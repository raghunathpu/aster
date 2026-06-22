import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1536, "height": 864})
        print("Navigating to http://localhost:8501...")
        await page.goto("http://localhost:8501", wait_until="domcontentloaded")
        
        # Wait for Streamlit to load the first page (Overview)
        print("Waiting for Overview page to load...")
        await page.wait_for_selector('h1:has-text("ASTER Intelligence")', timeout=30000)
        await page.wait_for_timeout(2000) # Give extra time for PyDeck map and weather widget
        await page.screenshot(path="screenshot_overview.png")
        print("Saved screenshot_overview.png")

        # Click on "Predict & Respond" in sidebar
        print("Clicking Predict & Respond tab...")
        await page.click('label:has-text("🔮 Predict & Respond")')
        await page.wait_for_selector('h1:has-text("Event Triage & Response Planner")', timeout=10000)
        await page.wait_for_timeout(2000)
        await page.screenshot(path="screenshot_predict_form.png")
        print("Saved screenshot_predict_form.png")
        
        # Click the "Analyse Event & Generate Response Plan" button
        print("Clicking Analyse button...")
        await page.click('button:has-text("Analyse Event & Generate Response Plan")')
        
        # Wait for predictions to render
        print("Waiting for predictions to render...")
        await page.wait_for_selector('h3:has-text("Forecasting & Triage Results")', timeout=30000)
        await page.wait_for_timeout(3000) # Give extra time for map and all elements to finish
        await page.screenshot(path="screenshot_predict_results.png", full_page=True)
        print("Saved screenshot_predict_results.png")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
