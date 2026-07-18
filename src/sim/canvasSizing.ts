export interface CanvasSize {
  width: number;
  height: number;
}

export function fitCanvasSize(
  cssWidth: number,
  cssHeight: number,
  scale: number,
  maximumWidth: number,
  maximumHeight: number,
): CanvasSize {
  const safeWidth = Math.max(1, Number.isFinite(cssWidth) ? cssWidth : 1);
  const safeHeight = Math.max(1, Number.isFinite(cssHeight) ? cssHeight : 1);
  const safeScale = Math.max(0.01, Number.isFinite(scale) ? scale : 1);
  const wantedWidth = safeWidth * safeScale;
  const wantedHeight = safeHeight * safeScale;
  const cap = Math.min(1, maximumWidth / wantedWidth, maximumHeight / wantedHeight);
  return {
    width: Math.max(1, Math.round(wantedWidth * cap)),
    height: Math.max(1, Math.round(wantedHeight * cap)),
  };
}
