export interface RenderTarget {
  texture: WebGLTexture;
  framebuffer: WebGLFramebuffer;
}

export class PingPongTarget {
  read: RenderTarget;
  write: RenderTarget;

  constructor(first: RenderTarget, second: RenderTarget) {
    this.read = first;
    this.write = second;
  }

  swap(): void {
    const previousRead = this.read;
    this.read = this.write;
    this.write = previousRead;
  }
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Unable to allocate a WebGL shader.');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? 'Unknown shader compiler error.';
    gl.deleteShader(shader);
    throw new Error(log);
  }
  return shader;
}

export class GLProgram {
  readonly handle: WebGLProgram;
  private readonly locations = new Map<string, WebGLUniformLocation | null>();

  constructor(
    private readonly gl: WebGL2RenderingContext,
    vertexSource: string,
    fragmentSource: string,
    transformFeedbackVaryings?: readonly string[],
  ) {
    const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
    const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    const program = gl.createProgram();
    if (!program) {
      gl.deleteShader(vertex);
      gl.deleteShader(fragment);
      throw new Error('Unable to allocate a WebGL program.');
    }
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    if (transformFeedbackVaryings?.length) {
      gl.transformFeedbackVaryings(program, [...transformFeedbackVaryings], gl.INTERLEAVED_ATTRIBS);
    }
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) ?? 'Unknown shader linker error.';
      gl.deleteProgram(program);
      throw new Error(log);
    }
    this.handle = program;
  }

  use(): void {
    this.gl.useProgram(this.handle);
  }

  set1f(name: string, value: number): void {
    this.gl.uniform1f(this.location(name), value);
  }

  set1i(name: string, value: number): void {
    this.gl.uniform1i(this.location(name), value);
  }

  set2i(name: string, x: number, y: number): void {
    this.gl.uniform2i(this.location(name), x, y);
  }

  set2f(name: string, x: number, y: number): void {
    this.gl.uniform2f(this.location(name), x, y);
  }

  dispose(): void {
    this.gl.deleteProgram(this.handle);
    this.locations.clear();
  }

  private location(name: string): WebGLUniformLocation | null {
    if (!this.locations.has(name)) {
      this.locations.set(name, this.gl.getUniformLocation(this.handle, name));
    }
    return this.locations.get(name) ?? null;
  }
}

export function createHalfFloatTarget(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
): RenderTarget {
  const texture = gl.createTexture();
  const framebuffer = gl.createFramebuffer();
  if (!texture || !framebuffer) {
    if (texture) gl.deleteTexture(texture);
    if (framebuffer) gl.deleteFramebuffer(framebuffer);
    throw new Error('Unable to allocate a half-float simulation target.');
  }

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, width, height);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(framebuffer);
    throw new Error('RGBA16F framebuffer is incomplete.');
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return { texture, framebuffer };
}

export function deleteTarget(gl: WebGL2RenderingContext, target: RenderTarget): void {
  gl.deleteFramebuffer(target.framebuffer);
  gl.deleteTexture(target.texture);
}

export function bindTextureUnit(
  gl: WebGL2RenderingContext,
  texture: WebGLTexture,
  unit: number,
): void {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, texture);
}
