import asyncio
import os
import shutil
from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from ollama import chat
import json
import re
import csv

from pyserxng.models import SearchConfig
from pyserxng import SearXNGClient
from pyserxng.models import InstanceInfo
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import html2text
import os
import unicodedata

# ==============================
# 1. Search ESG Score Titles
# ==============================
class SearchESGScoreBySearXNG:
    def __init__(self, url="http://localhost:8888"):
        self.client = SearXNGClient()
        self.instance = InstanceInfo(url=url)
        self.config = SearchConfig()

    def search_esg_score(self, company_name: str):
        query = f"site:spglobal.com {company_name} ESG Score"
        results = self.client.search(query, instance=self.instance, config=self.config)
        return results


# ==============================
# 2. Ollama Title Selector
# ==============================
class OllamaTitleSelector:
    def __init__(self, model_name="qwen2.5:3b"):
        self.model_name = model_name

    def safe_json_loads(self, raw: str):
        """
        Remove markdown ```json wrappers and parse JSON safely
        """
        raw = raw.strip()

        # remove markdown fences
        raw = re.sub(r"^```json", "", raw)
        raw = re.sub(r"^```", "", raw)
        raw = re.sub(r"```$", "", raw)

        return json.loads(raw.strip())

    def title_selector(self, company_name: str, titles_list: list[str]):

        system_prompt = f"""
        You are a title matching expert.
        Good titles should closely match the company's ESG report page: <company_name> ESG Score.

        Company name:
        {company_name}

        Titles list:
        {titles_list}

        Task:
        Select ONLY ONE best matching title.

        Output STRICT JSON:
        {{
          "index": <integer>,
          "title": "<string>",
          "reason": "<short reason>"
        }}
        """

        response = chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Pick the best title."},
            ],
            options={
                "temperature": 0.0,
                "num_ctx": 4096,
                "num_predict": 128,
            },
        )

        raw = response.message.content.strip()

        return self.safe_json_loads(raw)


# ==============================
# Helper: Clean Titles
# ==============================
def clean_title(t: str):
    """
    Convert "LasertecCorporationESGScore" → "Lasertec Corporation ESG Score"
    """
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)
    return t.strip()

def save_to_csv(data_row, csv_filename="esg_results.csv", is_first=False):
    """
    Lưu dữ liệu vào file CSV (hỗ trợ lưu dần dần)
    data_row: dict với các key: STT, Company Name, URL List, Titles List, Best Title, Best URL, Result ESG
    is_first: True nếu là lần đầu tạo file (để ghi header)
    """
    if not data_row:
        return
    
    fieldnames = ['STT', 'Company Name', 'URL List', 'Titles List', 'Best Title', 'Best URL', 'Result ESG']
    
    try:
        mode = 'w' if is_first else 'a'
        file_exists = Path(csv_filename).exists()
        
        with open(csv_filename, mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # Chỉ ghi header nếu là file mới
            if is_first or not file_exists:
                writer.writeheader()
            writer.writerow(data_row)
        print(f"✅ Đã lưu công ty '{data_row['Company Name']}' vào: {csv_filename}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file CSV: {e}")

# --- CẤU HÌNH ---
URL = "https://www.spglobal.com/sustainable1/en/scores/results?cid=4250712"
SOURCE_DIR = Path("mullvadData")
SOURCE_SAVE = Path("mullvadData_esgScore")

# --- HÀM HỖ TRỢ ---
def clear_folder(folder_path):
    """Xóa sạch nội dung bên trong folder."""
    if folder_path.exists():
        for item in folder_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        folder_path.mkdir(parents=True, exist_ok=True)
    print(f"🧹 Cleaned: {folder_path}")

def get_esg_markdown(html_path):
    """Parse dữ liệu từ file HTML lưu theo kiểu View Source (có thẻ span line)"""
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # 1. Thu thập tất cả text trong các span line để tái tạo lại nội dung HTML thực
        # Vì view-source chia nhỏ các thẻ thành nhiều span, ta cần gộp chúng lại thành 1 chuỗi lớn
        all_lines = soup.find_all('span', id=lambda x: x and x.startswith('line'))
        full_raw_text = "".join([line.get_text() for line in all_lines])
        
        # 2. Parse lại chuỗi text đó một lần nữa để nó trở thành HTML chuẩn
        clean_soup = BeautifulSoup(full_raw_text, 'html.parser')

        # 3. Tìm table mobile trong HTML đã làm sạch
        table = clean_soup.find('div', class_=lambda x: x and 'esg-table-mobile-container' in x)

        if not table:
            return "❌ Không tìm thấy bảng ESG trong nội dung View Source."

        # 4. Trích xuất Headers và Values
        headers = [h.get_text(strip=True) for h in table.find_all('div', attrs={'role': 'columnheader'})]
        values = [v.get_text(strip=True) for v in table.find_all('div', attrs={'role': 'cell'})]

        if not headers or not values:
            return "❌ Tìm thấy bảng nhưng dữ liệu bên trong trống."

        # 5. Chuẩn hóa dữ liệu và tạo Markdown
        md = "| Field | Value |\n| :--- | :--- |\n"
        for k, v in zip(headers, values):
            # Khử ký tự lạ như &amp; hoặc khoảng trắng đặc biệt
            clean_k = unicodedata.normalize("NFKD", k).replace('&amp;', '&')
            clean_v = unicodedata.normalize("NFKD", v).replace('&amp;', '&')
            md += f"| **{clean_k}** | {clean_v} |\n"

        return md

    except Exception as e:
        return f"❌ Lỗi xử lý file View Source: {e}"

# --- LOGIC CHÍNH ---
async def main(target_url):
    # 0. Dọn dẹp trước khi chạy
    clear_folder(SOURCE_DIR)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        # Sử dụng context ẩn danh (Private mode)
        context = await browser.new_context()
        page = await context.new_page()

        print(f"🚀 Đang kết nối noVNC...")
        await page.goto("http://localhost:3000/")
        await page.wait_for_timeout(5000)

        # 1. Tương tác với Browser bên trong noVNC
        canvas = page.locator("#videoCanvas")
        await canvas.click(force=True)

        # Focus address bar (Ctrl+L) và nhập URL
        await page.keyboard.press("Control+L")
        await page.keyboard.press("Control+A")
        await page.keyboard.press("Backspace")
        await page.keyboard.type(f"view-source:{target_url}", delay=20)
        await page.keyboard.press("Enter")

        print("⏳ Đang đợi trang tải dữ liệu (10s)...")
        await page.wait_for_timeout(10000) 

        # 2. Lưu trang (Ctrl + S)
        print("💾 Đang thực hiện lưu trang...")
        await page.keyboard.press("Control+S")
        await page.wait_for_timeout(2000)
        await page.keyboard.press("Enter")
        
        # Đợi một chút để file kịp ghi xuống đĩa
        await page.wait_for_timeout(2000)
        await browser.close()

    # 3. Xử lý file đã tải về
    print("\n--- KẾT QUẢ TRÍCH XUẤT ---")
    html_files = list(SOURCE_DIR.glob("*.html"))
    esg_result = ""
    
    if not html_files:
        print("⚠️ Không tìm thấy file HTML nào được lưu.")
        esg_result = "No HTML files found"
    else:
        for html_file in html_files:
            markdown_result = get_esg_markdown(html_file)
            print(f"\n📄 Nguồn: {html_file.name}")
            print(markdown_result)
            esg_result = markdown_result

    # 4. Dọn dẹp sau khi xong
    # clear_folder(SOURCE_DIR)
    
    return esg_result 

if __name__ == "__main__":
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
    
    searchESGScore = SearchESGScoreBySearXNG()
    selector = OllamaTitleSelector()
    csv_filename = "esg_results.csv"
    
    # # Xóa file CSV cũ nếu tồn tại
    # if Path(csv_filename).exists():
    #     Path(csv_filename).unlink()

    # for stt, name in enumerate(companies, 1):
    key = 5
    name = companies[key]
    stt = key + 1
    results = searchESGScore.search_esg_score(name)
    if not results.results or results.results[0].content == 'Sorry!':
        print(f"❌ No search results found for company '{name}'!")
        print("SearXNG Error")
        # Lưu dòng lỗi vào CSV ngay
        row_data = {
            'STT': stt,
            'Company Name': name,
            'URL List': '',
            'Titles List': '',
            'Best Title': '',
            'Best URL': '',
            'Result ESG': 'No search results found'
        }
        save_to_csv(row_data, csv_filename, is_first=(stt == 1))
        # continue

    TITLES = {i: r.title for i, r in enumerate(results.results)}
    # titles_list = [clean_title(t) for t in TITLES.values()]
    titles_list = [t for t in TITLES.values()]

    url_list = [str(r.url) for r in results.results]

    best = selector.title_selector(
        company_name=name,
        titles_list=titles_list
    )

    best_url = str(results.results[best['index']].url)
    best_title = best['title']
    # best_title = titles_list[0]
    # best_url = str(url_list[0])

    print(f"\n🔗 URL for company '{name}':")
    print(best_url)

    # Chạy main để lấy kết quả ESG
    esg_result = asyncio.run(main(best_url))
    
    # Lưu dữ liệu vào CSV ngay
    row_data = {
        'STT': stt,
        'Company Name': name,
        'URL List': '|'.join(url_list),  # Nối các URL bằng |
        'Titles List': '|'.join(titles_list),  # Nối các title bằng |
        'Best Title': best_title,
        'Best URL': best_url,
        'Result ESG': esg_result
    }
    save_to_csv(row_data, csv_filename, is_first=(stt == 1))
