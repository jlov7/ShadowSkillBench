import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./workbench.css";

export const metadata: Metadata = {
  title: "ShadowSkillBench Workbench",
  description: "Read-only research and custody workbench.",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
