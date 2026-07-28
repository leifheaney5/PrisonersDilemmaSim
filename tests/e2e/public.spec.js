const { test, expect } = require("@playwright/test");

const publicRoutes = [
  ["/", "Prisoner’s Dilemma Simulation"],
  ["/learn", "Why cooperation is difficult"],
  ["/profiles", "Strategy landscape"],
  ["/experiment", "Run Prisoner’s Dilemma experiments"],
];

for (const [route, heading] of publicRoutes) {
  test(`${route} renders without browser errors`, async ({ page }) => {
    const errors = [];
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    page.on("pageerror", (error) => errors.push(error.message));
    await page.goto(route, { waitUntil: "networkidle" });
    await expect(page.getByText(heading, { exact: false }).first()).toBeVisible();
    expect(errors).toEqual([]);
  });
}

test("donation page discloses third-party payment processing", async ({ page }) => {
  await page.route("https://www.paypal.com/**", (route) => route.abort());
  await page.goto("/donate", { waitUntil: "domcontentloaded" });
  await expect(page.getByText("Support the project", { exact: true })).toBeVisible();
  await expect(page.getByText("PayPal processes payment information", { exact: false })).toBeVisible();
});

test("learn payoff controls update the explanation", async ({ page }) => {
  await page.goto("/learn", { waitUntil: "networkidle" });
  await page.locator("#learn-player-move").click();
  await page.getByText("Defect", { exact: true }).last().click();
  await expect(page.getByText("You exploited the opponent", { exact: false })).toBeVisible();
  await expect(page.getByText("5", { exact: true }).first()).toBeVisible();
});

test("profiles render the 60-strategy landscape and profile selector", async ({ page }) => {
  await page.goto("/profiles", { waitUntil: "networkidle" });
  await expect(page.locator("#strategy-landscape .js-plotly-plot")).toBeVisible();
  await expect(page.locator("#profile-strategy")).toBeVisible();
  await expect(page.locator("#matchup-arena-round")).toBeVisible();
  await expect(page.locator("#matchup-arena-frame")).toBeVisible();
  await expect(page.getByText("See all 60 strategies at once", { exact: false })).toBeVisible();
});

test("experiment workspace exposes all primary modes", async ({ page }) => {
  await page.goto("/experiment", { waitUntil: "networkidle" });
  for (const label of ["Tournament (live)", "Play a match", "Build a strategy", "Robustness", "Evolution"]) {
    await expect(page.getByText(label, { exact: true }).first()).toBeVisible();
  }
  await expect(page.locator("#tournament-start")).toBeVisible();
  await page.getByText("Robustness", { exact: true }).first().click();
  await expect(page.locator("#robustness-ranking-metric")).toBeVisible();
  await expect(page.locator("#robustness-ranking-chart")).toBeVisible();
  await page.getByText("Evolution", { exact: true }).first().click();
  await expect(page.locator("#evolution-generation-ranking")).toBeVisible();
});

test("service endpoints and public security headers are available", async ({ request }) => {
  const ready = await request.get("/health/ready");
  expect(ready.ok()).toBeTruthy();
  expect((await ready.json()).status).toBe("ready");

  const home = await request.get("/");
  expect(home.headers()["x-content-type-options"]).toBe("nosniff");
  expect(home.headers()["content-security-policy"]).toContain("object-src 'none'");
  expect(home.headers()["x-request-id"]).toBeTruthy();
});
