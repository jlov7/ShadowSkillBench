import { expect, test } from "@playwright/test";

test("read-only overview loads with its HOLD and unavailable-data state", async ({ page }) => {
  await page.goto("/");

  await expect(page).toHaveTitle("ShadowSkillBench Workbench");
  await expect(page.getByRole("heading", { name: "Instruction hierarchy under contaminated skills" })).toBeVisible();
  await expect(page.getByText("HOLD — pending human anchor")).toBeVisible();
  await expect(page.getByText("No confirmatory data available.")).toBeVisible();
  await expect(page.getByRole("link", { name: "Export data table" }).first()).not.toHaveAttribute("download");
  await expect(page.getByLabel("Protocol custody")).toBeVisible();
});

test("episode compare uses the exact public practice IDs", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Episodes/ }).click();
  await expect(page.getByRole("heading", { name: "Episode compare" })).toBeVisible();
  await expect(page.getByText("practice-episode-A1").first()).toBeVisible();
  await expect(page.getByText("practice-episode-A3").first()).toBeVisible();
  await expect(page.getByText(/No prompts, chain-of-thought, raw provider payloads/i)).toBeVisible();
  await page.screenshot({ path: test.info().outputPath("episode-compare-1440x1000.png") });
});
