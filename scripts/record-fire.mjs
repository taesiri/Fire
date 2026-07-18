#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { access, mkdir, mkdtemp, open, readdir, rename, rm, stat, statfs, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import {
  FIELD_IDS,
  OUTPUT_PRESETS,
  buildCaptureConfig,
  createDryRunPlan,
  estimateOutputBytes,
  formatBytes,
  formatDuration,
  isHeavyJob,
  parseArguments,
  resolveOutputPath,
} from './record-fire-lib.mjs';

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(scriptDirectory, '..');
const sourceArguments = process.argv.slice(2);
let activeSession = null;
let viteServer = null;
let shuttingDown = false;
const browserTerminationPromises = new WeakMap();

function printHelp() {
  process.stdout.write(`
Fire Replica deterministic CLI recorder

Usage:
  Windows:  .\scripts\record-fire.cmd [options]
  Direct:   node scripts/record-fire.mjs [options]

Core options:
  --scene inferno                 methane | large-fire | inferno | rich | top-jet | base-spray
  --field beauty                 Repeat or comma-separate fields; use all for every field
  --duration 30s                 Exact 60 fps boundary, from one frame through 1h
  --resolution hd                hd/1080p | qhd/1440p | uhd/4k
  --solver-tier hd               hd | qhd | uhd; independent from output resolution
  --grid auto                    Named grid or exact XxYxZ (for example reference or 508x508x508)
  --memory-budget 32GiB          Exact aggregate allocation guard, up to 32 GiB
  --optical-detail 1024          0 | 256 | 512 | 1024 render-only frequency
  --output renders/fire.mp4      Exact path; multi-field jobs add the field name
  --output-dir renders           Used when --output is omitted

Execution and verification:
  --dry-run                      Print frames, disk, and GPU estimates without opening a browser
  --verify auto                  auto | metadata | sample | full
  --headless                     Experimental; headed Chrome is safer for hardware WebGPU
  --background                   Keep headed Chrome off-screen while preserving hardware WebGPU
  --browser PATH                 Override installed Chrome/Edge discovery
  --overwrite                    Replace only after the new file passes verification
  --confirm-heavy                Required for >=10 min, 4K >=1 min, or grids >=384 on any axis

Shortcuts:
  --test-views                   Four independent 30 s 1080p60 field-view acceptance jobs
  --quality highest              4K + UHD solver + real Reference 508³ + 32 GiB guard
  --hour                         Set duration to one hour
  --list-fields                  Print stable field IDs

Examples:
  .\scripts\record-fire.cmd --test-views --background
  .\scripts\record-fire.cmd --duration 30s --resolution 4k --solver-tier uhd --grid auto --field beauty
  .\scripts\record-fire.cmd --quality highest --hour --field beauty --confirm-heavy --background --output renders/firex-4k60-1h.mp4

The final example is expected to produce about 29.25 GB at 65 Mb/s. Reference 508³ is
a real ~13.19 GiB dense solver allocation and can fail on a 16 GiB adapter even when
the WebGPU structural limits pass. No grid, resolution, frame rate, or duration is
silently reduced.
`);
}

function printDryRun(options) {
  const plans = createDryRunPlan(options, repositoryRoot);
  process.stdout.write('\nDeterministic recording plan (no GPU work started)\n');
  for (const plan of plans) {
    process.stdout.write(`\n  Field:          ${plan.field}\n`);
    process.stdout.write(`  Output:         ${plan.output}\n`);
    process.stdout.write(`  Video:          ${plan.resolution} @ 60 fps, ${formatDuration(plan.durationSeconds)}, ${plan.frames.toLocaleString('en-US')} frames\n`);
    process.stdout.write(`  Solver:         ${plan.solverTier}, exact ${plan.grid} (${plan.gridCells.toLocaleString('en-US')} cells)\n`);
    process.stdout.write(`  GPU estimate:   ${formatBytes(plan.estimatedGpuBytes)} persistent solver resources\n`);
    process.stdout.write(`  Video estimate: ${formatBytes(plan.estimatedVideoBytes)} at the preset bitrate\n`);
  }
  process.stdout.write('\nThis streams encoded fMP4 payloads to disk; raw frames and the full master are never retained in browser RAM.\n');
}

function createDeferred() {
  let resolve;
  let reject;
  const promise = new Promise((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

function createCliSinkPlugin() {
  return {
    name: 'fire-replica-cli-sink',
    enforce: 'pre',
    configureServer(server) {
      server.middlewares.use(async (request, response, next) => {
        const requestUrl = new URL(request.url ?? '/', 'http://127.0.0.1');
        if (!requestUrl.pathname.startsWith('/__fire_cli/')) {
          next();
          return;
        }
        response.setHeader('cache-control', 'no-store');
        const session = activeSession;
        if (!session || requestUrl.searchParams.get('token') !== session.token) {
          sendText(response, 403, 'Invalid or inactive recording token.');
          return;
        }

        try {
          if (requestUrl.pathname === '/__fire_cli/config' && request.method === 'GET') {
            sendJson(response, 200, session.config);
            return;
          }
          if (requestUrl.pathname === '/__fire_cli/chunk' && request.method === 'POST') {
            await receiveChunk(session, requestUrl, request);
            response.statusCode = 204;
            response.setHeader('x-fire-cli-next-position', String(session.nextPosition));
            response.end();
            return;
          }
          if (requestUrl.pathname === '/__fire_cli/event' && request.method === 'POST') {
            const event = await readJsonBody(request, 1024 * 1024);
            handleBrowserEvent(session, event);
            response.statusCode = 204;
            response.end();
            return;
          }
          sendText(response, 404, 'Unknown recording endpoint.');
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          if (!session.settled) {
            session.settled = true;
            session.deferred.reject(error);
          }
          sendText(response, 500, message);
        }
      });
    },
  };
}

async function receiveChunk(session, requestUrl, request) {
  const position = Number(requestUrl.searchParams.get('position'));
  if (!Number.isSafeInteger(position) || position < 0) throw new Error('Chunk position must be a non-negative integer.');
  if (session.chunkInFlight) throw new Error('The browser attempted concurrent MP4 writes.');
  if (position !== session.nextPosition) {
    throw new Error(`Non-monotonic sink write at ${position}; expected ${session.nextPosition}.`);
  }
  session.chunkInFlight = true;
  try {
    let cursor = position;
    let requestBytes = 0;
    for await (const bodyChunk of request) {
      const bytes = bodyChunk instanceof Uint8Array ? bodyChunk : Buffer.from(bodyChunk);
      let offset = 0;
      while (offset < bytes.byteLength) {
        const result = await session.fileHandle.write(bytes, offset, bytes.byteLength - offset, cursor);
        if (result.bytesWritten <= 0) throw new Error('The local file sink stopped accepting bytes.');
        offset += result.bytesWritten;
        cursor += result.bytesWritten;
        requestBytes += result.bytesWritten;
      }
    }
    if (requestBytes === 0) throw new Error('The browser sent an empty MP4 chunk.');
    session.nextPosition = cursor;
  } finally {
    session.chunkInFlight = false;
  }
}

function handleBrowserEvent(session, event) {
  if (!event || typeof event !== 'object') throw new Error('Browser event must be a JSON object.');
  if (event.type === 'progress') {
    const completed = Number(event.completedFrames) || 0;
    const total = Number(event.totalFrames) || session.expectedFrames;
    const wall = Math.max(Number(event.wallSeconds) || 0, 0.001);
    const rate = completed / wall;
    const etaSeconds = rate > 0 ? Math.max(0, total - completed) / rate : 0;
    const percent = total > 0 ? Math.min(100, (completed / total) * 100) : 0;
    process.stdout.write(
      `\r  ${session.field.padEnd(16)} ${percent.toFixed(1).padStart(5)}%  ${completed.toLocaleString('en-US')} / ${total.toLocaleString('en-US')} frames  ${rate.toFixed(2)} render fps  ETA ${formatDuration(etaSeconds)}  ${formatBytes(session.nextPosition)} streamed   `,
    );
    return;
  }
  if (event.type === 'complete') {
    const frameCount = Number(event.frameCount);
    const streamedBytes = Number(event.streamedBytes);
    if (frameCount !== session.expectedFrames) {
      throw new Error(`Browser finalized ${frameCount} frames; expected ${session.expectedFrames}.`);
    }
    if (streamedBytes !== session.nextPosition) {
      throw new Error(`Browser reported ${streamedBytes} bytes; the sink committed ${session.nextPosition}.`);
    }
    session.browserResult = event;
    if (!session.settled) {
      session.settled = true;
      session.deferred.resolve(event);
    }
    process.stdout.write('\n');
    const performance = event.performance;
    if (performance && Number(performance.measuredFrames) > 0) {
      process.stdout.write(
        `  Average phases: ${Number(performance.averageSimulationMilliseconds).toFixed(1)} ms solver, ${Number(performance.averagePresentationMilliseconds ?? performance.averagePresentationReadbackMilliseconds).toFixed(1)} ms presentation, ${Number(performance.averageVideoFrameConstructionMilliseconds).toFixed(1)} ms VideoFrame.\n`,
      );
    }
    return;
  }
  if (event.type === 'error') {
    throw new Error(`Browser capture failed: ${String(event.message ?? 'unknown error')}`);
  }
  throw new Error(`Unknown browser event type: ${String(event.type)}.`);
}

async function readJsonBody(request, maximumBytes) {
  const chunks = [];
  let bytes = 0;
  for await (const chunk of request) {
    bytes += chunk.byteLength;
    if (bytes > maximumBytes) throw new Error('Browser event body is too large.');
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

function sendJson(response, status, value) {
  const body = JSON.stringify(value);
  response.statusCode = status;
  response.setHeader('content-type', 'application/json; charset=utf-8');
  response.setHeader('content-length', Buffer.byteLength(body));
  response.end(body);
}

function sendText(response, status, value) {
  response.statusCode = status;
  response.setHeader('content-type', 'text/plain; charset=utf-8');
  response.end(value);
}

async function startCaptureServer() {
  const server = await createServer({
    root: repositoryRoot,
    configFile: path.join(repositoryRoot, 'vite.config.ts'),
    logLevel: 'error',
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    plugins: [createCliSinkPlugin()],
  });
  await server.listen();
  const address = server.httpServer?.address();
  if (!address || typeof address === 'string') throw new Error('Vite did not expose a local TCP capture address.');
  return { server, origin: `http://127.0.0.1:${address.port}` };
}

async function findBrowser(explicitPath) {
  const candidates = [
    explicitPath,
    process.env.CHROME_PATH,
    process.env.EDGE_PATH,
    process.env.PROGRAMFILES && path.join(process.env.PROGRAMFILES, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    process.env['PROGRAMFILES(X86)'] && path.join(process.env['PROGRAMFILES(X86)'], 'Google', 'Chrome', 'Application', 'chrome.exe'),
    process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, 'Google', 'Chrome', 'Application', 'chrome.exe'),
    process.env.PROGRAMFILES && path.join(process.env.PROGRAMFILES, 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    process.env['PROGRAMFILES(X86)'] && path.join(process.env['PROGRAMFILES(X86)'], 'Microsoft', 'Edge', 'Application', 'msedge.exe'),
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return path.resolve(candidate);
    } catch {
      // Try the next installed browser path.
    }
  }
  throw new Error('Chrome or Edge was not found. Pass its executable path with --browser.');
}

function launchBrowser(executable, url, profileDirectory, options) {
  const argumentsList = [
    `--user-data-dir=${profileDirectory}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    '--disable-background-timer-throttling',
    '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows',
    '--disable-features=CalculateNativeWinOcclusion',
    '--autoplay-policy=no-user-gesture-required',
    '--window-size=1280,800',
  ];
  if (options.headless) argumentsList.push('--headless=new', '--enable-unsafe-webgpu');
  if (options.background && !options.headless) argumentsList.push('--window-position=-32000,-32000');
  argumentsList.push(url);
  return spawn(executable, argumentsList, { stdio: ['ignore', 'ignore', 'pipe'], windowsHide: options.background || options.headless });
}

async function ensureDiskSpace(outputPath, estimatedBytes) {
  const directory = path.dirname(outputPath);
  await mkdir(directory, { recursive: true });
  const filesystem = await statfs(directory);
  const available = Number(filesystem.bavail) * Number(filesystem.bsize);
  const required = Math.ceil(estimatedBytes * 1.15 + 1024 ** 3);
  if (available < required) {
    throw new Error(`Only ${formatBytes(available)} is free near ${directory}; this job requires at least ${formatBytes(required)}.`);
  }
  return available;
}

async function pathExists(filePath) {
  try {
    await stat(filePath);
    return true;
  } catch (error) {
    if (error?.code === 'ENOENT') return false;
    throw error;
  }
}

function partialPathFor(outputPath) {
  const extension = path.extname(outputPath);
  return extension ? `${outputPath.slice(0, -extension.length)}.partial${extension}` : `${outputPath}.partial.mp4`;
}

async function runRecordingJob({ options, field, browserPath, origin }) {
  const outputPath = resolveOutputPath(options, field, repositoryRoot);
  if (path.extname(outputPath).toLowerCase() !== '.mp4') throw new Error(`Output must end in .mp4: ${outputPath}`);
  const partialPath = partialPathFor(outputPath);
  const manifestPath = `${outputPath}.json`;
  const partialManifestPath = `${manifestPath}.partial`;
  const estimatedBytes = estimateOutputBytes(options.outputPreset, options.durationSeconds);
  const availableBytes = await ensureDiskSpace(outputPath, estimatedBytes);

  for (const candidate of [outputPath, manifestPath]) {
    if (await pathExists(candidate)) {
      if (!options.overwrite) throw new Error(`${candidate} already exists. Use --overwrite to replace it.`);
    }
  }
  for (const candidate of [partialPath, partialManifestPath]) {
    if (!(await pathExists(candidate))) continue;
    if (!options.overwrite) throw new Error(`${candidate} already exists. Use --overwrite to replace it.`);
    await rm(candidate, { force: true });
  }

  const token = randomBytes(24).toString('hex');
  const jobId = `${options.scene}-${field}-${Date.now().toString(36)}`;
  const config = buildCaptureConfig(options, field, jobId);
  const deferred = createDeferred();
  const fileHandle = await open(partialPath, 'w+');
  const profileDirectory = await mkdtemp(path.join(os.tmpdir(), 'fire-replica-capture-'));
  const startedAt = new Date();
  const expectedFrames = Math.round(options.durationSeconds * 60);
  const session = {
    token,
    config,
    field,
    expectedFrames,
    nextPosition: 0,
    chunkInFlight: false,
    fileHandle,
    deferred,
    settled: false,
    browserResult: null,
    browserProcess: null,
    partialPath,
  };
  activeSession = session;

  process.stdout.write(`\nRecording ${field} -> ${outputPath}\n`);
  process.stdout.write(`  ${OUTPUT_PRESETS[options.outputPreset].width}x${OUTPUT_PRESETS[options.outputPreset].height} @ 60 fps, ${expectedFrames.toLocaleString('en-US')} frames\n`);
  process.stdout.write(`  Solver ${options.solverTier}, exact ${options.effectiveDimensions.join('x')} grid, ${formatBytes(options.memoryEstimate.totalBytes)} estimated GPU resources\n`);
  process.stdout.write(`  ${formatBytes(availableBytes)} output-disk free (not VRAM); approximately ${formatBytes(estimatedBytes)} video payload expected\n`);
  if (options.memoryEstimate.totalBytes >= 8 * 1024 ** 3) {
    process.stdout.write('  Heavy grid must fit on one WebGPU adapter; close other GPU-heavy apps and tabs before capture.\n');
  }

  const captureUrl = `${origin}/capture.html?fire-cli=${encodeURIComponent(token)}`;
  const browserProcess = launchBrowser(browserPath, captureUrl, profileDirectory, options);
  session.browserProcess = browserProcess;
  let browserStderr = '';
  browserProcess.stderr?.setEncoding('utf8');
  browserProcess.stderr?.on('data', (chunk) => {
    browserStderr = `${browserStderr}${chunk}`.slice(-12_000);
  });
  browserProcess.once('error', (error) => {
    if (!session.settled) {
      session.settled = true;
      session.deferred.reject(new Error(`Browser could not be started: ${error.message}`, { cause: error }));
    }
  });
  browserProcess.once('exit', (code, signal) => {
    if (!session.settled) {
      session.settled = true;
      session.deferred.reject(new Error(
        `Browser exited before capture finalized (code ${String(code)}, signal ${String(signal)}).${browserStderr ? `\n${browserStderr}` : ''}`,
      ));
    }
  });

  try {
    const browserResult = await deferred.promise;
    await fileHandle.sync();
    await fileHandle.close();
    session.fileClosed = true;
    await terminateBrowser(browserProcess);

    process.stdout.write('  Browser stream finalized. Verifying from disk...\n');
    const verification = await verifyRecording(partialPath, options, expectedFrames);
    const finishedAt = new Date();
    const gitCommit = await optionalCommand('git', ['rev-parse', 'HEAD'], repositoryRoot);
    const browserVersion = await discoverBrowserVersion(browserPath);
    const manifest = {
      schemaVersion: 1,
      status: 'verified',
      outputPath,
      startedAt: startedAt.toISOString(),
      finishedAt: finishedAt.toISOString(),
      wallSeconds: (finishedAt.getTime() - startedAt.getTime()) / 1000,
      repositoryCommit: gitCommit.trim() || null,
      browser: browserVersion,
      config,
      expectedFrames,
      estimatedVideoBytes: estimatedBytes,
      browserResult,
      verification,
    };
    await writeFile(partialManifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
    await installVerifiedPair([
      { sourcePath: partialPath, destinationPath: outputPath },
      { sourcePath: partialManifestPath, destinationPath: manifestPath },
    ], token);
    process.stdout.write(`  Verified ${verification.width}x${verification.height}, ${verification.frameCount.toLocaleString('en-US')} frames, ${verification.frameRate.toFixed(3)} fps, ${formatDuration(verification.durationSeconds)}.\n`);
    process.stdout.write(`  Saved ${outputPath}\n  Manifest ${manifestPath}\n`);
    return { outputPath, manifestPath, verification };
  } catch (error) {
    let cleanupError = null;
    try {
      await terminateBrowser(browserProcess);
    } catch (terminationError) {
      cleanupError = terminationError;
    }
    const preserved = [];
    if (await pathExists(partialPath).catch(() => false)) preserved.push(`Unverified partial video: ${partialPath}`);
    if (await pathExists(partialManifestPath).catch(() => false)) preserved.push(`Pending manifest: ${partialManifestPath}`);
    if (await pathExists(outputPath).catch(() => false)) preserved.push(`Existing or installed verified video: ${outputPath}`);
    process.stdout.write('\n');
    throw new Error(
      `${error instanceof Error ? error.message : String(error)}${cleanupError ? `\nBrowser cleanup failed: ${cleanupError instanceof Error ? cleanupError.message : String(cleanupError)}` : ''}${preserved.length ? `\n${preserved.join('\n')}` : ''}`,
      { cause: error },
    );
  } finally {
    activeSession = null;
    if (!session.fileClosed) await fileHandle.close().catch(() => undefined);
    await terminateBrowser(browserProcess).catch((error) => {
      process.stderr.write(`Warning: browser cleanup remains incomplete: ${error instanceof Error ? error.message : String(error)}\n`);
    });
    try {
      await rm(profileDirectory, { recursive: true, force: true });
    } catch (error) {
      process.stderr.write(`Warning: could not remove temporary browser profile ${profileDirectory}: ${error instanceof Error ? error.message : String(error)}\n`);
    }
  }
}

async function installVerifiedPair(entries, token) {
  const states = entries.map(({ sourcePath, destinationPath }) => ({
    sourcePath,
    destinationPath,
    backupPath: `${destinationPath}.previous-${token.slice(0, 12)}`,
    backedUp: false,
    installed: false,
  }));
  try {
    for (const state of states) {
      if (!(await pathExists(state.destinationPath))) continue;
      await rename(state.destinationPath, state.backupPath);
      state.backedUp = true;
    }
    for (const state of states) {
      await rename(state.sourcePath, state.destinationPath);
      state.installed = true;
    }
  } catch (error) {
    const rollbackErrors = [];
    for (const state of [...states].reverse()) {
      if (!state.installed) continue;
      try {
        await rename(state.destinationPath, state.sourcePath);
      } catch (rollbackError) {
        rollbackErrors.push(`could not preserve ${state.destinationPath}: ${rollbackError instanceof Error ? rollbackError.message : String(rollbackError)}`);
      }
    }
    for (const state of [...states].reverse()) {
      if (!state.backedUp) continue;
      try {
        await rename(state.backupPath, state.destinationPath);
      } catch (rollbackError) {
        rollbackErrors.push(`could not restore ${state.destinationPath}: ${rollbackError instanceof Error ? rollbackError.message : String(rollbackError)}`);
      }
    }
    throw new Error(
      `Could not install the verified video/manifest pair: ${error instanceof Error ? error.message : String(error)}${rollbackErrors.length ? `; rollback errors: ${rollbackErrors.join('; ')}` : ''}`,
      { cause: error },
    );
  }
  for (const state of states) {
    if (!state.backedUp) continue;
    await rm(state.backupPath, { force: true }).catch((error) => {
      process.stderr.write(`Warning: installed the verified pair but could not remove backup ${state.backupPath}: ${error instanceof Error ? error.message : String(error)}\n`);
    });
  }
}

function waitForProcessExit(child, timeoutMilliseconds) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return Promise.resolve(true);
  return new Promise((resolve) => {
    let timer;
    const finish = (exited) => {
      clearTimeout(timer);
      child.off('exit', handleExit);
      child.off('error', handleError);
      resolve(exited);
    };
    const handleExit = () => finish(true);
    const handleError = () => finish(false);
    child.once('exit', handleExit);
    child.once('error', handleError);
    timer = setTimeout(() => finish(false), timeoutMilliseconds);
    timer.unref?.();
  });
}

function terminateBrowser(browserProcess) {
  if (!browserProcess) return Promise.resolve();
  const existing = browserTerminationPromises.get(browserProcess);
  if (existing) return existing;
  const termination = terminateBrowserOnce(browserProcess);
  browserTerminationPromises.set(browserProcess, termination);
  void termination.then(
    () => browserTerminationPromises.delete(browserProcess),
    () => browserTerminationPromises.delete(browserProcess),
  );
  return termination;
}

async function terminateBrowserOnce(browserProcess) {
  if (!browserProcess || browserProcess.exitCode !== null || browserProcess.signalCode !== null) return;
  if (!browserProcess.pid) return;
  if (process.platform === 'win32' && browserProcess.pid) {
    const killer = spawn('taskkill.exe', ['/PID', String(browserProcess.pid), '/T', '/F'], {
      stdio: 'ignore',
      windowsHide: true,
    });
    const taskkillFinished = await waitForProcessExit(killer, 10_000);
    const treeKillSucceeded = taskkillFinished && killer.exitCode === 0;
    let browserExited = await waitForProcessExit(browserProcess, 5_000);
    if (!browserExited) {
      try {
        browserProcess.kill('SIGKILL');
      } catch {
        // It may have exited between the timeout and the signal.
      }
      browserExited = await waitForProcessExit(browserProcess, 5_000);
    }
    // taskkill returns 128 when Chromium exits between our initial liveness
    // check and taskkill opening the PID. If the spawned root has definitely
    // emitted exit, cleanup succeeded and must not invalidate a finished video.
    if (!treeKillSucceeded && !browserExited) {
      throw new Error(`taskkill could not confirm termination of Chromium process tree ${browserProcess.pid} (exit ${String(killer.exitCode)}).`);
    }
    if (!browserExited) throw new Error(`Chromium process ${browserProcess.pid} remained alive after forced tree termination.`);
    return;
  } else {
    try {
      browserProcess.kill('SIGTERM');
    } catch {
      return;
    }
  }
  if (await waitForProcessExit(browserProcess, 5_000)) return;
  try {
    browserProcess.kill('SIGKILL');
  } catch {
    // It may have exited between the timeout and the signal.
  }
  if (!(await waitForProcessExit(browserProcess, 5_000))) {
    throw new Error(`Chromium process ${browserProcess.pid} remained alive after SIGKILL.`);
  }
}

async function verifyRecording(filePath, options, expectedFrames) {
  const preset = OUTPUT_PRESETS[options.outputPreset];
  const metadataText = await runCommand('ffprobe', [
    '-v', 'error',
    '-select_streams', 'v:0',
    '-count_frames',
    '-show_entries', 'stream=codec_name,profile,level,codec_tag_string,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_read_frames,nb_frames,duration,bit_rate:format=format_name,duration,size',
    '-of', 'json',
    filePath,
  ], { maximumOutputBytes: 8 * 1024 * 1024 });
  const metadata = JSON.parse(metadataText);
  const stream = metadata.streams?.[0];
  if (!stream) throw new Error('ffprobe found no video stream in the finalized file.');
  const width = Number(stream.width);
  const height = Number(stream.height);
  const frameRate = parseRational(stream.avg_frame_rate || stream.r_frame_rate);
  const frameCount = Number(stream.nb_read_frames || stream.nb_frames);
  const durationSeconds = Number(stream.duration || metadata.format?.duration);
  if (stream.codec_name !== 'h264') throw new Error(`ffprobe reported ${String(stream.codec_name)}, expected H.264.`);
  if (stream.codec_tag_string !== 'avc1') throw new Error(`ffprobe reported ${String(stream.codec_tag_string)}, expected avc1.`);
  if (stream.profile !== 'High') throw new Error(`ffprobe reported H.264 ${String(stream.profile)}, expected High profile.`);
  if (Number(stream.level) !== preset.level) {
    throw new Error(`ffprobe reported H.264 level ${String(stream.level)}, expected level ${preset.level}.`);
  }
  if (stream.pix_fmt !== 'yuv420p') throw new Error(`ffprobe reported ${String(stream.pix_fmt)}, expected yuv420p.`);
  if (width !== preset.width || height !== preset.height) {
    throw new Error(`ffprobe reported ${width}x${height}, expected ${preset.width}x${preset.height}.`);
  }
  if (!Number.isFinite(frameRate) || Math.abs(frameRate - 60) > 0.001) {
    throw new Error(`ffprobe reported ${frameRate} fps, expected exact 60 fps.`);
  }
  if (frameCount !== expectedFrames) throw new Error(`ffprobe counted ${frameCount} frames, expected ${expectedFrames}.`);
  const expectedDuration = expectedFrames / 60;
  if (!Number.isFinite(durationSeconds) || Math.abs(durationSeconds - expectedDuration) > 1 / 60 + 0.001) {
    throw new Error(`ffprobe reported ${durationSeconds}s, expected ${expectedDuration}s.`);
  }

  const verificationMode = options.verify === 'auto'
    ? options.durationSeconds <= 120 ? 'full' : 'sample'
    : options.verify;
  const cadence = await verifyPacketCadence(filePath, expectedFrames);
  if (verificationMode === 'full') {
    await decodeWindow(filePath, null, null);
  } else if (verificationMode === 'sample') {
    const sampleDuration = Math.min(3, expectedDuration);
    const starts = [...new Set([
      0,
      Math.max(0, expectedDuration / 2 - sampleDuration / 2),
      Math.max(0, expectedDuration - sampleDuration),
    ].map((value) => Number(value.toFixed(3))))];
    for (const start of starts) await decodeWindow(filePath, start, sampleDuration);
  }

  return {
    codec: stream.codec_name,
    codecTag: stream.codec_tag_string,
    profile: stream.profile ?? null,
    level: stream.level ?? null,
    pixelFormat: stream.pix_fmt ?? null,
    width,
    height,
    frameRate,
    frameCount,
    durationSeconds,
    bitRate: Number(stream.bit_rate) || null,
    fileBytes: Number(metadata.format?.size) || null,
    verificationMode,
    cadence,
  };
}

async function verifyPacketCadence(filePath, expectedFrames) {
  return new Promise((resolve, reject) => {
    const child = spawn('ffprobe', [
      '-v', 'error',
      '-select_streams', 'v:0',
      '-show_packets',
      '-show_entries', 'packet=pts_time,duration_time,flags',
      '-of', 'csv=p=0',
      filePath,
    ], { cwd: repositoryRoot, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    const lines = readline.createInterface({ input: child.stdout, crlfDelay: Infinity });
    const frameDuration = 1 / 60;
    let frameCount = 0;
    let keyFrameCount = 0;
    let maximumTimestampErrorSeconds = 0;
    let maximumDurationErrorSeconds = 0;
    let maximumKeyFrameGapSeconds = 0;
    let lastKeyFrameTimestamp = null;
    let parseError = null;
    let stderr = '';

    lines.on('line', (line) => {
      if (parseError || line.trim() === '') return;
      const [timestampValue, durationValue, flags = ''] = line.split(',');
      const timestamp = Number(timestampValue);
      const duration = Number(durationValue);
      if (!Number.isFinite(timestamp) || !Number.isFinite(duration)) {
        parseError = new Error(`Packet ${frameCount} has invalid timing: ${line}`);
        child.kill();
        return;
      }
      maximumTimestampErrorSeconds = Math.max(
        maximumTimestampErrorSeconds,
        Math.abs(timestamp - frameCount * frameDuration),
      );
      maximumDurationErrorSeconds = Math.max(
        maximumDurationErrorSeconds,
        Math.abs(duration - frameDuration),
      );
      if (flags.includes('K')) {
        keyFrameCount += 1;
        if (lastKeyFrameTimestamp !== null) {
          maximumKeyFrameGapSeconds = Math.max(maximumKeyFrameGapSeconds, timestamp - lastKeyFrameTimestamp);
        }
        lastKeyFrameTimestamp = timestamp;
      }
      frameCount += 1;
    });
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk) => {
      stderr = `${stderr}${chunk}`.slice(-1024 * 1024);
    });
    child.once('error', reject);
    child.once('close', (code, signal) => {
      lines.close();
      if (parseError) {
        reject(parseError);
        return;
      }
      if (code !== 0) {
        reject(new Error(`ffprobe cadence scan failed (code ${String(code)}, signal ${String(signal)}): ${stderr}`));
        return;
      }
      if (frameCount !== expectedFrames) {
        reject(new Error(`Cadence scan inspected ${frameCount} packets, expected ${expectedFrames}.`));
        return;
      }
      if (maximumTimestampErrorSeconds > 0.001) {
        reject(new Error(`Packet timestamps drift by up to ${(maximumTimestampErrorSeconds * 1000).toFixed(3)} ms.`));
        return;
      }
      if (maximumDurationErrorSeconds > 0.001) {
        reject(new Error(`Packet durations drift by up to ${(maximumDurationErrorSeconds * 1000).toFixed(3)} ms.`));
        return;
      }
      const minimumKeyFrames = Math.max(1, Math.ceil(expectedFrames / 60));
      if (keyFrameCount < minimumKeyFrames || maximumKeyFrameGapSeconds > 1.001) {
        reject(new Error(`Keyframe cadence failed: ${keyFrameCount} keyframes, maximum gap ${maximumKeyFrameGapSeconds.toFixed(3)}s.`));
        return;
      }
      resolve({
        frameCount,
        keyFrameCount,
        maximumKeyFrameGapSeconds,
        maximumTimestampErrorSeconds,
        maximumDurationErrorSeconds,
      });
    });
  });
}

async function decodeWindow(filePath, startSeconds, durationSeconds) {
  const argumentsList = ['-v', 'error'];
  if (startSeconds !== null) argumentsList.push('-ss', String(startSeconds));
  argumentsList.push('-i', filePath);
  if (durationSeconds !== null) argumentsList.push('-t', String(durationSeconds));
  argumentsList.push('-map', '0:v:0', '-an', '-f', 'null', process.platform === 'win32' ? 'NUL' : '/dev/null');
  await runCommand('ffmpeg', argumentsList, { maximumOutputBytes: 8 * 1024 * 1024 });
}

function parseRational(value) {
  const [numerator, denominator] = String(value).split('/').map(Number);
  return denominator ? numerator / denominator : Number(value);
}

async function runCommand(executable, argumentsList, { cwd = repositoryRoot, maximumOutputBytes = 16 * 1024 * 1024 } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, argumentsList, { cwd, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    const stdout = [];
    const stderr = [];
    let outputBytes = 0;
    const append = (collection, chunk) => {
      outputBytes += chunk.byteLength;
      if (outputBytes > maximumOutputBytes) {
        child.kill();
        reject(new Error(`${executable} produced more than ${formatBytes(maximumOutputBytes)} of diagnostic output.`));
        return;
      }
      collection.push(chunk);
    };
    child.stdout.on('data', (chunk) => append(stdout, chunk));
    child.stderr.on('data', (chunk) => append(stderr, chunk));
    child.once('error', reject);
    child.once('close', (code, signal) => {
      const standardOutput = Buffer.concat(stdout).toString('utf8');
      const standardError = Buffer.concat(stderr).toString('utf8');
      if (code === 0) resolve(standardOutput);
      else reject(new Error(`${executable} failed (code ${String(code)}, signal ${String(signal)}): ${standardError || standardOutput}`));
    });
  });
}

async function optionalCommand(executable, argumentsList, cwd) {
  try {
    return await runCommand(executable, argumentsList, { cwd, maximumOutputBytes: 1024 * 1024 });
  } catch {
    return '';
  }
}

async function discoverBrowserVersion(browserPath) {
  try {
    const entries = await readdir(path.dirname(browserPath), { withFileTypes: true });
    const versions = entries
      .filter((entry) => entry.isDirectory() && /^\d+(?:\.\d+){2,}$/.test(entry.name))
      .map((entry) => entry.name)
      .sort((left, right) => right.localeCompare(left, undefined, { numeric: true }));
    if (versions[0]) return `${path.basename(browserPath)} ${versions[0]}`;
  } catch {
    // The executable name remains sufficient provenance when version metadata is unavailable.
  }
  return path.basename(browserPath);
}

async function main() {
  const options = parseArguments(sourceArguments);
  if (options.help) {
    printHelp();
    return;
  }
  if (options.listFields) {
    process.stdout.write(`${FIELD_IDS.join('\n')}\n`);
    return;
  }
  if (options.dryRun) {
    printDryRun(options);
    return;
  }
  if (isHeavyJob(options) && !options.confirmHeavy) {
    printDryRun(options);
    throw new Error('This is a heavy-duty request. Review the plan, then rerun with --confirm-heavy.');
  }

  const browserPath = await findBrowser(options.browser);
  const { server, origin } = await startCaptureServer();
  viteServer = server;
  process.stdout.write(`Using ${browserPath}\nCapture service ${origin} (loopback only)\n`);
  const results = [];
  for (const field of options.fields) {
    results.push(await runRecordingJob({ options, field, browserPath, origin }));
  }
  process.stdout.write(`\nCompleted ${results.length} verified recording${results.length === 1 ? '' : 's'}.\n`);
}

process.on('SIGINT', () => {
  if (shuttingDown) return;
  shuttingDown = true;
  process.stdout.write('\nStopping capture; the partial file will be preserved.\n');
  if (activeSession && !activeSession.settled) {
    activeSession.settled = true;
    activeSession.deferred.reject(new Error('Capture interrupted by the user.'));
    void terminateBrowser(activeSession.browserProcess).catch((error) => {
      process.stderr.write(`Browser cleanup failed: ${error instanceof Error ? error.message : String(error)}\n`);
    });
  }
});

try {
  await main();
} catch (error) {
  process.stderr.write(`\nRecording failed: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
} finally {
  if (viteServer) await viteServer.close().catch(() => undefined);
}
