from searcher_searxng import SearXNGSearch
from validator_llm import SearchReportValidator
from search_on_jpx import JPXGovernanceScraper
from datetime import datetime
import csv
import os
from search_on_company_site import normalize_domain

class SearchReportCombine:
    def __init__(self,headless=True):
        self.xng_searcher = SearXNGSearch()
        self.validator = SearchReportValidator()
        self.jpx_scraper = JPXGovernanceScraper(headless=True)

    def parse_date(self, date_str):
        if not date_str:
            return None
        formats = [
            "%Y/%m/%d",
            "%Y-%m-%d",
            "%Y.%m.%d",
            "%Y/%m",
            "%Y"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    def normalize_report(self, source_name, data):
        if not data:
            return None

        if "date" in data:
            date_value = data.get("date")
            url = data.get("pdf_url")
        else:
            date_value = data.get("detected_date")
            url = data.get("url")

        parsed = self.parse_date(date_value)

        if not parsed or not url:
            return None

        return {
            "source": source_name,
            "url": url,
            "date": parsed,
            "raw_date": date_value,
        }
    def select_latest(self, *reports):
        valid_reports = [r for r in reports if r is not None]

        if not valid_reports:
            return None

        return max(valid_reports, key=lambda x: x["date"])

    async def __call__(self, stock_id: str, company_name: str, company_site: str):
        # 1 - Search on company site
        query_on_company_site = f"site:{normalize_domain(company_site)} filetype:pdf corporate governance report"
        results_on_company_site = self.xng_searcher.search(query_on_company_site)
        best_on_company_site = self.validator.best_report(query_on_company_site, results_on_company_site)

        # 2 - Search on nikkei site
        query_on_nikkei_site = f"site:www.nikkei.com/markets/ir/irftp/data/tdnr/tdnetg3 filetype:pdf CORPORATE GOVERNANCE 最終更新日 {company_name}"
        results_on_nikkei_site = self.xng_searcher.search(query_on_nikkei_site)
        best_on_nikkei_site = self.validator.best_report(query_on_nikkei_site, results_on_nikkei_site)

        # 3 - Search on JPX - Listed Company Search
        async with self.jpx_scraper as scraper:
            best_on_jpx = await scraper.get_latest_governance(stock_id)

        normalize_company = self.normalize_report("company_site", best_on_company_site)
        normalize_nikkei = self.normalize_report("nikkei_site", best_on_nikkei_site)
        normalize_jpx = self.normalize_report("jpx_site", best_on_jpx)

        latest = self.select_latest(normalize_company, normalize_nikkei, normalize_jpx)

        return latest

    async def process_companies(self, companies: dict, output_file: str = "governance_reports_combined.csv"):
        """
        Process list of companies and save results to CSV file.
        
        Args:
            companies: dict with stock_id as key and company_info dict as value
                {
                    "7203": {
                        "name": "TOYOTA MOTOR CORPORATION",
                        "site": "global.toyota"
                    },
                    ...
                }
            output_file: output CSV file path
        """
        
        # Write header if file doesn't exist
        file_exists = os.path.exists(output_file)
        
        with open(output_file, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Stock ID", "Company Name", "Source", "URL", "Date", "Raw Date"])
            
            for stock_id, company_info in companies.items():
                print(f"Processing {company_info['name']} ({stock_id})")
                
                try:
                    result = await self(
                        stock_id=stock_id,
                        company_name=company_info['name'],
                        company_site=company_info['site']
                    )
                    
                    if result:
                        writer.writerow([
                            stock_id,
                            company_info['name'],
                            result['source'],
                            result['url'],
                            result['date'].strftime("%Y-%m-%d"),
                            result['raw_date'],
                        ])
                    else:
                        writer.writerow([
                            stock_id,
                            company_info['name'],
                            None,
                            None,
                            None,
                            None,
                        ])
                    
                    f.flush()  # Ensure data is written immediately
                    
                except Exception as e:
                    print(f"❌ Error processing {company_info['name']} ({stock_id}): {e}")
                    writer.writerow([
                        stock_id,
                        company_info['name'],
                        None,
                        None,
                        None,
                        None,
                    ])
                    f.flush()
        
        print(f"\n✅ Done. Saved to: {os.path.abspath(output_file)}")