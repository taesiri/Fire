import { cp, mkdir, readdir, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const siteRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const clientDist = resolve(siteRoot, ".client-dist");
const publicRoot = resolve(siteRoot, "public");

await mkdir(publicRoot, { recursive: true });
await rm(join(publicRoot, "assets"), { recursive: true, force: true });
await rm(join(publicRoot, "lab.html"), { force: true });

for (const entry of await readdir(clientDist, { withFileTypes: true })) {
  const source = join(clientDist, entry.name);
  const target = join(
    publicRoot,
    entry.name === "index.html" ? "lab.html" : entry.name,
  );
  await cp(source, target, { recursive: entry.isDirectory(), force: true });
}
