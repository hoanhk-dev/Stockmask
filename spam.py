import asyncio
from playwright.async_api import async_playwright
import pyperclip
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


SEARCH_URL = "http://localhost:3000"
COMPANY_NAME = "Lasertec Corporation"

import google.generativeai as genai
import json
import re


class SearchResultExtractor:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name

    def run_gemini(self, prompt: str) -> str:
        model = genai.GenerativeModel(model_name=self.model_name)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        )

        return response.text.strip()

    def normalize_url(self, url: str) -> str:
        url = url.strip()

        if "›" in url:
            url = url.replace("›", "/")
            url = re.sub(r"\s+", "", url)

        return url

    # ============================
    # STEP 1: Extract results
    # ============================
    def extract_results(self, copied_text: str) -> dict:
        prompt = f"""
You are given copied text from a search engine results page.

Extract all search results.

Each result must include:
- title
- url

Return ONLY JSON:

{{
  "results": [
    {{
      "title": "...",
      "url": "..."
    }}
  ]
}}

Rules:
- Ignore ads/navigation.
- URLs may appear in breadcrumb form with › symbols.
- Output JSON only.

Copied text:
----------------
{copied_text}
----------------
"""

        raw_json = self.run_gemini(prompt)
        data = json.loads(raw_json)

        for r in data["results"]:
            r["url"] = self.normalize_url(r["url"])

        return data

    # ============================
    # STEP 2: Identify best match
    # ============================
    def rank_results(self, query: str, results: list) -> dict:
        prompt = f"""
You are given a search query and extracted search results.

Your job:

1. Identify which result best matches the query.
2. Mark results that truly belong to the query intent.
3. Especially prioritize results from spglobal.com.

Query:
{query}

Search Results:
{json.dumps(results, indent=2)}

Return ONLY JSON in this format:

{{
  "best_result": {{
    "title": "...",
    "url": "...",
    "reason": "why this is best"
  }},
  "matched_results": [
    {{
      "title": "...",
      "url": "...",
      "match_reason": "..."
    }}
  ]
}}

Rules:
- best_result must be one of the given results.
- matched_results should include only relevant ones.
- Prefer spglobal.com ESG Score page if present.
"""

        raw_json = self.run_gemini(prompt)
        return json.loads(raw_json)

async def search_and_get_result(page, company_name):
    query = f"site:spglobal.com {company_name} ESG Score"
    print(f"🔍 Searching: {query}")

    canvas = page.locator("#videoCanvas")
    await canvas.click(force=True)

    await page.keyboard.press("Control+L")
    await page.keyboard.press("Control+A")
    await page.keyboard.press("Backspace")

    await page.keyboard.type(query, delay=40)
    await page.keyboard.press("Enter")

    print("⏳ Waiting search results...")
    await asyncio.sleep(6)

    # Copy page text
    print("📋 Ctrl+A then Ctrl+C ...")
    await page.keyboard.press("Control+A")
    await page.keyboard.press("Control+C")
    await asyncio.sleep(1)

    copied_text = pyperclip.paste()

    # ======================
    # STEP 1 Extract raw
    # ======================
    extracted = extractor.extract_results(copied_text)

    # ======================
    # STEP 2 Rank + Filter
    # ======================
    ranked = extractor.rank_results(query, extracted["results"])

    print("\n🎯 FINAL JSON RESULT:")
    print(json.dumps(ranked, indent=2, ensure_ascii=False))

    return ranked

extractor = SearchResultExtractor()
    
async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        page = await browser.new_page()
        await page.goto(SEARCH_URL)
        await asyncio.sleep(5)
        print("✅ Browser ready...\n")

        result = await search_and_get_result(page, COMPANY_NAME)

        print("\n🎯 RESULT:")
        print(result)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())