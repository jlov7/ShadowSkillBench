import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import Page from "./page";

describe("static read-only workbench", () => {
  afterEach(() => cleanup());
  it("renders the committed HOLD overview with unavailable chart data and export tables", () => {
    render(<Page />);
    expect(screen.getByText("HOLD — pending human anchor")).not.toBeNull();
    expect(screen.getAllByText("Unavailable").length).toBeGreaterThan(4);
    expect(screen.getByRole("img", { name: /Completion-under-policy response curves.*no confirmatory Stage A report rows/i })).not.toBeNull();
    expect(screen.getAllByRole("link", { name: "Export data table" })[0].getAttribute("download")).toBeNull();
  });

  it("supports keyboard section navigation while ignoring input targets", () => {
    render(<Page />);
    fireEvent.keyDown(window, { key: "6" });
    expect(screen.getByRole("heading", { name: "Episode compare" })).not.toBeNull();
    const input = document.createElement("input");
    document.body.append(input);
    fireEvent.keyDown(input, { key: "1" });
    expect(screen.getByRole("heading", { name: "Episode compare" })).not.toBeNull();
    input.remove();
  });

  it("keeps practice annotations hidden until the explicit toggle and renders exactly 12 cards", () => {
    render(<Page />);
    fireEvent.click(screen.getByRole("button", { name: /Demonstrations/ }));
    expect(screen.getAllByText(/Worker /).length).toBe(12);
    expect(screen.queryByText(/Practice review annotation withheld/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Reveal benchmark ground truth" }));
    expect(screen.getAllByText(/Practice review annotation withheld/).length).toBe(12);
  });

  it("compares the exact sanitised fixture IDs without exposing private trace material", () => {
    render(<Page />);
    fireEvent.keyDown(window, { key: "6" });
    expect(screen.getAllByText("practice-episode-A1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("practice-episode-A3").length).toBeGreaterThan(0);
    expect(screen.getByText(/No prompts, chain-of-thought, raw provider payloads, oracle truth/i)).not.toBeNull();
  });

  it("ships no API route or server-action mutation surface", () => {
    const apiDirectory = resolve(process.cwd(), "app/api");
    const pageSource = readFileSync(resolve(process.cwd(), "app/page.tsx"), "utf8");

    expect(existsSync(apiDirectory)).toBe(false);
    expect(pageSource).not.toContain('"use server"');
  });
});
