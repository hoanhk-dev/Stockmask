import csv
import time

def nikkei_governance_search(
    searcher,
    validator,
    companies,
    output_file="nikkei_governance_best_results.csv",
    delay=10
):
    """
    Search Nikkei governance PDF for a list of companies
    and save best validated result to CSV.
    """

    # ========================
    # CREATE CSV HEADER
    # ========================
    with open(output_file, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "company_name",
            "title",
            "url",
            "category",
            "detected_date",
            "why_best"
        ])

    # ========================
    # MAIN LOOP
    # ========================
    for stock_id, info in companies.items():
        name = info["name"]
        print("\n" + "=" * 100)
        print("Company:", name)
        print("=" * 100)

        query = f"""
        site:www.nikkei.com/markets/ir/irftp/data/tdnr/tdnetg3 CORPORATE GOVERNANCE 最終更新日 {name} filetype:pdf
        """

        print("Query:", query.strip())

        try:
            search_results = searcher.search(query)

            best = validator.best_report(query, search_results)

            if not best:
                print("⚠ No valid best result detected")
                continue

            print("Best:", best)

            with open(output_file, mode="a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    name,
                    best.get("title"),
                    best.get("url"),
                    best.get("category"),
                    best.get("detected_date"),
                    best.get("why_best")
                ])

            print("✅ Saved best report:", best.get("title"))

        except Exception as e:
            print(f"❌ Error processing {name}: {e}")

        time.sleep(delay)

    print("\nDONE. File saved:", output_file)