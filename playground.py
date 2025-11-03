from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # show the browser
    page = browser.new_page()
    page.goto("https://google.com")

    print("Page title:", page.title())

    page.screenshot(path="example.png")
    #browser.close()


