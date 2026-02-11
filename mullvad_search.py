# docker run -d \
#   --name mullvad-browser \
#   --platform=linux/amd64 \
#   -p 3000:3000 \
#   -p 3001:3001 \
#   -e TZ=Asia/Ho_Chi_Minh \
#   -v "$(pwd)/mullvadData:/config/Downloads" \
#   --shm-size=1gb \
#   lscr.io/linuxserver/mullvad-browser:latest

import asyncio
import re
import pyperclip
from playwright.async_api import async_playwright

SEARCH_URL = "http://localhost:3000"

async def searchEngine(query: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # ===== Open search engine =====
        await page.goto(SEARCH_URL)
        await asyncio.sleep(5)

        # ===== Search =====
        print(f"🔍 Searching: {query}")
        await page.keyboard.press("Control+L")
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Backspace")
        await page.keyboard.type(query, delay=20)
        await page.keyboard.press("Enter")

        print("⏳ Waiting for results...")
        await asyncio.sleep(5)

        # ===== Copy results with retry =====
        max_copy_retries = 3
        copy_retry = 0
        raw_text = ""
        
        while copy_retry < max_copy_retries:
            print("📋 Copying results...")
            await page.keyboard.press("Control+A")
            await page.keyboard.press("Control+C")
            await asyncio.sleep(1)

            raw_text = pyperclip.paste()
            
            if raw_text and len(raw_text.strip()) > 10:
                print(f"✅ Got clipboard content ({len(raw_text)} chars)")
                break
            else:
                copy_retry += 1
                if copy_retry < max_copy_retries:
                    print(f"⚠️ Clipboard empty/too short, retrying ({copy_retry}/{max_copy_retries - 1})...")
                    await asyncio.sleep(2)
                else:
                    print(f"❌ Failed to get clipboard after {max_copy_retries} attempts")
        
        await browser.close()

    # ===== PARSE RESULTS =====
    def normalize_url(url: str) -> str:
        url = re.sub(r"\s*›\s*", "/", url)
        url = re.sub(r"/{2,}", "/", url)
        url = url.replace("https:/", "https://")
        return url.strip()

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    print(f"📄 Total lines after split: {len(lines)}")
    
    results = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if re.search(r"https?://", line):
            url = normalize_url(line)
            title = None
            content = None

            j = i + 1
            while j < len(lines):
                if not re.search(r"https?://", lines[j]):
                    if title is None:
                        title = lines[j]
                    elif content is None:
                        content = lines[j]
                        break
                j += 1

            if title:
                results.append({
                    "title": title,
                    "url": url,
                    "content": content
                })

            i = j
        else:
            i += 1

    return results

async def main():
    query = "site:r-i.co.jp Code: 5020 credit rating"
    results = await searchEngine(query)

    print("\n" + "="*80)
    for idx, r in enumerate(results, 1):
        print(f"\n[{idx}] {r['title']}")
        print(f"    URL: {r['url']}")
        if r['content']:
            print(f"    Content: {r['content']}")
    print("\n" + "="*80)
    print(f"Tổng cộng: {len(results)} kết quả\n")


asyncio.run(main())