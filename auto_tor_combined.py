import asyncio
import os
import shutil
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import pyperclip
import google.generativeai as genai
import json
import re
import csv
import unicodedata
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# --- CẤU HÌNH ---
SEARCH_URL = "http://localhost:3000"
SOURCE_DIR = Path("mullvadData")
CSV_FILENAME = "esg_results.csv"


# ==============================
# 1. Gemini Search & URL Selector (optimized - 1 call)
# ==============================
class SearchAndSelectURL:
    """Extract search results and select best URL in ONE Gemini call"""
    
    def __init__(self, model_name="gemini-2.5-flash"):
        self.model_name = model_name

    def run_gemini(self, prompt: str) -> str:
        try:
            model = genai.GenerativeModel(model_name=self.model_name)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
            )
            if not response or not hasattr(response, 'text') or not response.text:
                print(f"⚠️ Empty response from Gemini")
                return '{"title": "", "url": "", "all_results": []}'
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ Gemini API Error: {e}")
            return '{"title": "", "url": "", "all_results": []}'

    def normalize_url(self, url: str) -> str:
        url = url.strip()
        if "›" in url:
            url = url.replace("›", "/")
            url = re.sub(r"\s+", "", url)
        return url

    def select_best_url(self, company_name: str, copied_text: str) -> dict:
        """
        Extract results AND select best URL in ONE call.
        Returns: {
            "title": "...",
            "url": "...",
            "all_results": [{"title": "...", "url": "..."}, ...]
        }
        """
        if not copied_text or copied_text.strip() == "":
            print("⚠️ No search results copied from browser")
            return {"title": "", "url": "", "all_results": []}
        
        prompt = f"""
        You are given copied text from a search engine results page.

        Company: {company_name}
        Query: site:spglobal.com {company_name} ESG Score

        Task:
        1. Extract ALL search results (title + url)
        2. Identify the SINGLE BEST result matching the ESG Score page for this company
        3. Prioritize spglobal.com results

        Return ONLY JSON:
        {{
        "title": "<best result title>",
        "url": "<best result url>",
        "all_results": [
            {{"title": "...", "url": "..."}},
            ...
        ]
        }}

        Rules:
        - Ignore ads/navigation
        - URLs may have › symbols (convert to /)
        - best_url should be the URL of the top ESG score result
        - Include ALL extracted results in all_results array

        Copied text:
        ----------------
        {copied_text}
        ----------------
        """
        try:
            raw_json = self.run_gemini(prompt)
            if not raw_json or raw_json.strip() == "":
                return {"title": "", "url": "", "all_results": []}
            
            data = json.loads(raw_json)

            # Normalize URLs
            if data.get("url"):
                data["url"] = self.normalize_url(data["url"])
            for r in data.get("all_results", []):
                if r.get("url"):
                    r["url"] = self.normalize_url(r["url"])

            return data
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON Parse Error: {e}")
            print(f"   Raw response: {raw_json[:200]}")
            return {"title": "", "url": "", "all_results": []}
        except Exception as e:
            print(f"⚠️ Error in select_best_url: {e}")
            return {"title": "", "url": "", "all_results": []}


# ==============================
# 2. Helper functions
# ==============================
def clear_folder(folder_path):
    """Clear folder contents"""
    if folder_path.exists():
        for item in folder_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        folder_path.mkdir(parents=True, exist_ok=True)
    print(f"🧹 Cleaned: {folder_path}")


def get_esg_markdown(html_content):
    """Parse ESG data from HTML content string (view-source format)"""
    try:
        # Remove view-source HTML markers if present (line numbers, etc)
        html_content = re.sub(r'<span[^>]*id="line\d+"[^>]*>.*?</span>', '', html_content, flags=re.DOTALL)
        html_content = re.sub(r'line\d+:', '', html_content)
        
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the ESG detail table component container
        table_container = soup.find('div', class_=lambda x: x and 'esg-detail-table-component' in x)
        
        if not table_container:
            return "❌ No esg-detail-table-component found."

        # Try desktop version first (esg-table-container)
        desktop_table = table_container.find('div', class_=lambda x: x and 'esg-table' in x and 'mobile' not in (x or ''))
        
        if desktop_table:
            # Desktop version: structure with rowgroups
            rowgroups = desktop_table.find_all('div', role='rowgroup')
            if len(rowgroups) >= 2:
                # Extract headers
                header_row = rowgroups[0].find('div', role='row')
                headers = [h.get_text(strip=True) for h in header_row.find_all('div', role='columnheader')] if header_row else []
                
                # Extract values
                data_row = rowgroups[1].find('div', role='row')
                values = [v.get_text(strip=True) for v in data_row.find_all('div', role='cell')] if data_row else []
                
                if headers and values:
                    md = "| Field | Value |\n| :--- | :--- |\n"
                    for k, v in zip(headers, values):
                        clean_k = unicodedata.normalize("NFKD", k).replace('&amp;', '&')
                        clean_v = unicodedata.normalize("NFKD", v).replace('&amp;', '&')
                        md += f"| **{clean_k}** | {clean_v} |\n"
                    return md

        # Try mobile version (esg-table-mobile-container)
        mobile_table = table_container.find('div', class_=lambda x: x and 'esg-table-mobile-container' in x)
        
        if mobile_table:
            # Mobile version: alternating headers and cells
            headers = []
            values = []
            
            all_divs = mobile_table.find_all('div', role=['columnheader', 'cell'])
            for div in all_divs:
                role = div.get('role', '')
                text = div.get_text(strip=True)
                if role == 'columnheader':
                    headers.append(text)
                elif role == 'cell' and text:  # Avoid empty cells
                    values.append(text)
            
            if headers and values:
                # Mobile: pair headers with first set of values (first row)
                md = "| Field | Value |\n| :--- | :--- |\n"
                for k, v in zip(headers, values[:len(headers)]):
                    clean_k = unicodedata.normalize("NFKD", k).replace('&amp;', '&')
                    clean_v = unicodedata.normalize("NFKD", v).replace('&amp;', '&')
                    md += f"| **{clean_k}** | {clean_v} |\n"
                return md

        return "❌ Could not parse ESG table from HTML."

    except Exception as e:
        return f"❌ Error processing HTML: {e}"


def save_to_csv(data_row, csv_filename=CSV_FILENAME, is_first=False):
    """Save data to CSV file"""
    if not data_row:
        return
    
    fieldnames = ['STT', 'Company Name', 'URL List', 'Best Title', 'Best URL', 'Result ESG']
    
    try:
        mode = 'w' if is_first else 'a'
        file_exists = Path(csv_filename).exists()
        
        with open(csv_filename, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if is_first or not file_exists:
                writer.writeheader()
            writer.writerow(data_row)
        print(f"✅ Saved '{data_row['Company Name']}' to: {csv_filename}")
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")


def get_processed_stt_from_csv(csv_filename=CSV_FILENAME):
    """Get list of STT numbers already processed in CSV"""
    processed_stt = set()
    
    if not Path(csv_filename).exists():
        return processed_stt
    
    try:
        with open(csv_filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('STT'):
                    try:
                        stt = int(row['STT'])
                        processed_stt.add(stt)
                    except ValueError:
                        pass
    except Exception as e:
        print(f"⚠️ Error reading CSV: {e}")
    
    return processed_stt


# ==============================
# MAIN WORKFLOW - Browser opened ONCE
# ==============================
async def process_company(stt, company_name, page):
    """
    Process single company using existing browser page
    
    Args:
        stt: Sequential number
        company_name: Company name to search
        page: Playwright page object (already open)
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        if retry_count > 0:
            print(f"\n🔄 Retry attempt {retry_count}/{max_retries - 1}...")
        
        print(f"\n{'='*60}")
        print(f"[{stt}] {company_name}")
        if retry_count > 0:
            print(f"(Attempt {retry_count + 1}/{max_retries})")
        print(f"{'='*60}")

        selector = SearchAndSelectURL()
        query = f"site:spglobal.com {company_name} ESG Score"

        # ===== STEP 1: Search =====
        print(f"🔍 Searching: {query}")
        await page.keyboard.press("Control+L")
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Backspace")
        await page.keyboard.type(query, delay=40)
        await page.keyboard.press("Enter")

        print("⏳ Waiting for results...")
        await asyncio.sleep(6)

        # ===== STEP 2: Copy & Extract results in ONE call =====
        print("📋 Copying & analyzing results...")
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Control+C")
        await asyncio.sleep(1)

        copied_text = pyperclip.paste()
        
        if not copied_text:
            print(f"❌ Failed to copy text - ({len(copied_text) if copied_text else 0} chars)")
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(3)  # Wait before retry
                continue
            else:
                # Max retries reached, save error
                row_data = {
                    'STT': stt,
                    'Company Name': company_name,
                    'URL List': '',
                    'Best Title': '',
                    'Best URL': '',
                    'Result ESG': 'Clipboard empty - search failed (max retries)'
                }
                save_to_csv(row_data, is_first=(stt == 1))
                return
        
        # Single Gemini call to extract AND select best URL
        try:
            result = selector.select_best_url(company_name, copied_text)
        except Exception as e:
            print(f"❌ Error processing search results: {e}")
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(3)  # Wait before retry
                continue
            else:
                row_data = {
                    'STT': stt,
                    'Company Name': company_name,
                    'URL List': '',
                    'Best Title': '',
                    'Best URL': '',
                    'Result ESG': f'Error: {str(e)} (max retries)'
                }
                save_to_csv(row_data, is_first=(stt == 1))
                return

        best_title = result.get("title", "")
        best_url = result.get("url", "")
        all_urls = [r["url"] for r in result.get("all_results", []) if r.get("url")]

        if not best_url:
            print(f"❌ No ESG URL found for '{company_name}'!")
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(3)  # Wait before retry
                continue
            else:
                row_data = {
                    'STT': stt,
                    'Company Name': company_name,
                    'URL List': '|'.join(all_urls),
                    'Best Title': '',
                    'Best URL': '',
                    'Result ESG': 'No ESG URL found (max retries)'
                }
                save_to_csv(row_data, is_first=(stt == 1))
                return

        print(f"✅ Selected: {best_title}")
        print(f"🔗 URL: {best_url}")

        # ===== STEP 3: Extract ESG data (copy from view-source) =====
        print("📊 Extracting ESG data...")
        
        # Navigate to view-source using the SAME page
        await page.keyboard.press("Control+L")
        await asyncio.sleep(0.5)  # Wait for URL bar to be ready
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Backspace")
        
        # Type view-source URL with proper delays
        view_source_url = f"view-source:{best_url}"
        print(f"📝 Typing: {view_source_url}")
        await page.keyboard.type(view_source_url, delay=30)
        await asyncio.sleep(1)  # Wait for full URL to be typed
        await page.keyboard.press("Enter")

        print("⏳ Loading view-source page...")
        await asyncio.sleep(12)  # Longer wait for view-source to render
        print("Clearing clipboard and copying page content...")
        pyperclip.copy("")
        await asyncio.sleep(0.5)
        # Copy page content from clipboard
        print("📋 Copying page content...")
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Control+C")
        await asyncio.sleep(1)

        html_content = pyperclip.paste()
        
        if not html_content:
            print(f"⚠️ Clipboard content too short ({len(html_content) if html_content else 0} chars)")
            esg_result = "Clipboard content too short"
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(3)  # Wait before retry
                continue
            # If max retries reached, save with current result
        else:
            # Process clipboard content directly
            markdown_result = get_esg_markdown(html_content)
            print(f"\n📄 ESG Data:\n{markdown_result[:200]}...")
            esg_result = markdown_result

        # ===== STEP 4: Save to CSV =====
        row_data = {
            'STT': stt,
            'Company Name': company_name,
            'URL List': '|'.join(all_urls),
            'Best Title': best_title,
            'Best URL': best_url,
            'Result ESG': esg_result
        }
        save_to_csv(row_data, is_first=(stt == 1))
        
        # Success - break out of retry loop
        if esg_result != "No HTML files found":
            print(f"✨ Successfully processed [{stt}] {company_name}")
            return
        else:
            # HTML not found, but retry
            retry_count += 1
            if retry_count < max_retries:
                print(f"⚠️ HTML extraction failed, retrying...")
                await asyncio.sleep(3)
    
    # All retries exhausted
    print(f"⚠️ Max retries reached for [{stt}] {company_name}")


async def main():
    """Main entry point - opens browser ONCE for searching"""
    companies = [
        "Lasertec Corporation",
        "Shin-Etsu Chemical Co., Ltd.",
        "Mitsui O.S.K. Lines, Ltd.",
        "DeNA Co., Ltd.",
        "CAPCOM CO., LTD.",
        "BRIDGESTONE CORPORATION",
        "DAIICHI SANKYO COMPANY, LIMITED",
        "KAJIMA CORPORATION",
        "JFE Holdings, Inc.",
        "Nintendo Co., Ltd.",
        "Mitsubishi Heavy Industries, Ltd.",
        "ITOCHU Corporation",
        "CASIO COMPUTER CO., LTD.",
        "Bank of Innovation, Inc.",
        "Mitsubishi UFJ Financial Group, Inc.",
        "ADVANTEST CORPORATION",
        "TOYOTA MOTOR CORPORATION",
        "FAST RETAILING CO., LTD.",
        "Mitsui Fudosan Co., Ltd.",
        "SoftBank Group Corp.",
        "KOEI TECMO HOLDINGS CO., LTD.",
        "TOEI ANIMATION CO., LTD.",
        "ENEOS Holdings, Inc.",
        "Mitsubishi Corporation",
        "LY Corporation",
    ]

    # Get already processed STT numbers
    processed_stt = get_processed_stt_from_csv()
    
    if processed_stt:
        print(f"📋 Already processed: {sorted(processed_stt)}")
        print(f"⏭️  Skipping these companies...\n")

    # ===== Open search browser ONCE =====
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(SEARCH_URL)
        await asyncio.sleep(5)
        print("✅ Browser ready for searching\n")

        # Process ALL companies using the same page
        for stt, name in enumerate(companies, 1):
            # Skip if already processed
            if stt in processed_stt:
                print(f"\n⏭️  [{stt}] {name} - Already processed, skipping")
                continue
            
            await process_company(stt, name, page)
        
        await browser.close()

    print("\n✅ All done!")


if __name__ == "__main__":
    asyncio.run(main())
