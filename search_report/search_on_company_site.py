from urllib.parse import urlparse
import json
import time
import csv
import os

def normalize_domain(url: str) -> str:
    """
    Convert:
        https://www.lasertec.co.jp/ -> lasertec.co.jp
        https://laserTec.co.jp/en/ -> lasertec.co.jp
        http://www.mhi.com -> mhi.com
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain

def on_company_site_search(
    searcher,
    validator,
    company_info: dict,
    output_file: str,
    search_keyword: str,
    result_label: str,
    delay: int = 5
):
    """
    Generic search function.

    search_keyword: ví dụ
        "integrated report"
        "corporate governance report"
        "sustainability report"

    result_label: tên cột CSV
        "Integrated"
        "Governance"
    """

    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Company",
            f"{result_label} Results",
            f"{result_label} Best URL"
        ])

        for stock_id, info in company_info.items():
            domain = normalize_domain(info["site"])

            print("\n==============================")
            print(f"Processing domain: {domain}")
            print("==============================")

            best_url = ""
            results = []

            try:
                query = f"site:{domain} filetype:pdf {search_keyword}"
                results = searcher.search(query)

                if results:
                    best = validator.best_report(query, results)
                    best_url = best.get("url", "") if best else ""

            except Exception as e:
                print(f"Search failed: {domain}")
                print(e)

            results_json = json.dumps(results, ensure_ascii=False)

            writer.writerow([
                domain,
                results_json,
                best_url
            ])

            f.flush()
            time.sleep(delay)
            print(f"Saved: {domain}")

    print("\nDone. Saved to:", os.path.abspath(output_file))
