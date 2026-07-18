import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Fire Replica Lab",
  description:
    "Realtime WebGPU and WebGL fire simulation experiments inspired by two research papers.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#080b0d" }}>{children}</body>
    </html>
  );
}
