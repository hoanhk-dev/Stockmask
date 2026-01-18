STOCK_ID = [
    "50200",
    "80580",
    "69200",
    "80010",
    "69520",
    "43930",
    "91040",
    "79740",
    "72030",
    "24320",
    "46890",
    "40630",
    "51080",
    "96970",
    "99830",
    "99840",
    "45680",
    "18120",
    "70110",
    "88010",
    "36350",
    "48160",
    "83060",
    "54110",
    "68570"
]

def stock_id_preprocessing(stock_ids):
    result = []
    for stock in stock_ids:
        if stock.endswith("0"):
            stock = stock[:-1] + ".T"
        result.append(stock)
    return result

STOCK_ID = stock_id_preprocessing(STOCK_ID)
print(STOCK_ID)



from itertools import product
import yfinance as yf
import pandas as pd
import numpy as np

class FuzzyBase:
    """Shared fuzzy helper methods used by multiple fuzzy-system classes."""
    def in_range(self, x, a, b):
        return a < x < b

    def linear_fuzzy(self, x, x1, y1, x2, y2):
        """Calculate y value at x on the line connecting (x1, y1) and (x2, y2)."""
        if x1 == x2:
            return max(y1, y2)
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    def Rule(self, a, b):
        return self.RULE_TABLE.get((a, b), ("UNKNOWN", 0))

    def defuzzify_sugeno(self, combine_rules):
        numerator = 0.0
        denominator = 0.0
        for r in combine_rules:
            numerator += r["weight"] * r["score"]
            denominator += r["weight"]
        if denominator == 0:
            return 0
        return numerator / denominator

    def map_fuzzy_output_centroid(self, score):
        label = min(
            self.OUTPUT_LEVELS.items(),
            key=lambda x: abs(score - x[1])
        )
        return label

    def infer_rules(self, fuzzy1, fuzzy2, key1_name, key2_name):
        """Generic rule inference combining two fuzzy membership dicts.

        Returns a list of rule dictionaries with keys using key1_name and key2_name.
        """
        combine_rules = []
        for (l1, w1), (l2, w2) in product(fuzzy1.items(), fuzzy2.items()):
            rule_name, rule_score = self.Rule(l1, l2)
            firing_strength = min(w1, w2)
            combine_rules.append({
                key1_name: l1,
                key2_name: l2,
                "label": rule_name,
                "score": rule_score,
                "weight": firing_strength,
            })
        return combine_rules


class ROABottomingTrendFuzzy(FuzzyBase):
    def __init__(self, MAP_ROA, MAP_TREND, OUTPUT_LEVELS, RULE_TABLE):
        self.MAP_ROA = MAP_ROA
        self.MAP_TREND = MAP_TREND
        self.OUTPUT_LEVELS = OUTPUT_LEVELS
        self.RULE_TABLE = RULE_TABLE
    
    def fetch_roa_multi_year(self, stock_id):
        ticker = yf.Ticker(stock_id)
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet

        if financials is None or balance_sheet is None:
            return None
        if financials.empty or balance_sheet.empty:
            return None

        roa_by_year = {}

        for col in financials.columns:
            try:
                # ---- safe get ----
                if "Net Income" not in financials.index:
                    continue
                if "Total Assets" not in balance_sheet.index:
                    continue

                net_income = financials.loc["Net Income", col]
                total_assets = balance_sheet.loc["Total Assets", col]

                # ---- NaN / None check ----
                if pd.isna(net_income) or pd.isna(total_assets):
                    continue

                # ---- zero / invalid check ----
                if total_assets == 0:
                    continue

                roa = net_income / total_assets

                # ---- final NaN / inf guard ----
                if np.isnan(roa) or np.isinf(roa):
                    continue

                roa_by_year[col.year] = roa

            except Exception:
                continue

        if not roa_by_year:
            return None

        # sort by year
        return [roa_by_year[y] for y in sorted(roa_by_year)]
    def fuzzy_product(self, roa_value, slope):
        roa_raw = []
        trend_raw = []

        # ---------- ROA fuzzification ----------
        for key, (a, b) in self.MAP_ROA.items():
            if self.in_range(roa_value, a, b):
                roa_raw.append((key, a, b))

        ROA_FUZZY = {}
        if len(roa_raw) == 1:
            ROA_FUZZY[roa_raw[0][0]] = 1.0
        elif len(roa_raw) == 2:
            (k1, a1, b1), (k2, a2, b2) = roa_raw
            left = max(a1, a2)
            right = min(b1, b2)
            ROA_FUZZY[k1] = self.linear_fuzzy(roa_value, left, 1, right, 0)
            ROA_FUZZY[k2] = self.linear_fuzzy(roa_value, left, 0, right, 1)
        else:
            ROA_FUZZY["UNKNOWN"] = 0.0

        # ---------- TREND fuzzification ----------
        for key, (a, b) in self.MAP_TREND.items():
            if self.in_range(slope, a, b):
                trend_raw.append((key, a, b))

        TREND_FUZZY = {}
        if len(trend_raw) == 1:
            TREND_FUZZY[trend_raw[0][0]] = 1.0
        elif len(trend_raw) == 2:
            (k1, a1, b1), (k2, a2, b2) = trend_raw
            left = max(a1, a2)
            right = min(b1, b2)
            TREND_FUZZY[k1] = self.linear_fuzzy(slope, left, 1, right, 0)
            TREND_FUZZY[k2] = self.linear_fuzzy(slope, left, 0, right, 1)
        else:
            TREND_FUZZY["UNKNOWN"] = 0.0

        return ROA_FUZZY, TREND_FUZZY

    def slope_of_list(self, data):
        n = len(data)
        if n < 2:
            return 0.0

        x = np.arange(n)
        y = np.array(data)

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope
    
    def ROA_Bottoming_Trend(self, stock_id, get_level_label=False):
        data = self.fetch_roa_multi_year(stock_id)
        if not data:
            raise ValueError("No ROA data available for stock: {}".format(stock_id))
        min_value = min(data)
        min_index = data.index(min_value) 
        if min_index == len(data) - 1:
            # Left
            slope = self.slope_of_list(data)
        if min_index == 0:
            # Right
            slope = self.slope_of_list(data)
        if 0 < min_index < len(data) - 1:
            # Left
            slope = self.slope_of_list(data[min_index:])

        roe_finall_data = data[-1]
        roa_fuzzy, trend_fuzzy = self.fuzzy_product(roe_finall_data, slope)
        # print("ROA FUZZY:", roa_fuzzy)
        # print("TREND FUZZY:", trend_fuzzy)
        combine_rules = super().infer_rules(roa_fuzzy, trend_fuzzy, 'roa', 'trend')
        # print("COMBINE RULES:", combine_rules)

        final_score = self.defuzzify_sugeno(combine_rules)
        
        if get_level_label:
            level_label = self.map_fuzzy_output_centroid(final_score)
            return level_label, final_score
        return final_score


class AssetLiquidityFuzzy(FuzzyBase):
    def __init__(self, MAP_CURRENT_RATIO, MAP_QUICK_RATIO, OUTPUT_LEVELS, RULE_TABLE):
        self.MAP_CURRENT_RATIO = MAP_CURRENT_RATIO
        self.MAP_QUICK_RATIO = MAP_QUICK_RATIO
        self.OUTPUT_LEVELS = OUTPUT_LEVELS
        self.RULE_TABLE = RULE_TABLE

    def fetch_asset_liquidity(self, stock_id):
        ticker = yf.Ticker(stock_id)
        CurentRatio = ticker.balance_sheet.loc["Current Assets"] / ticker.balance_sheet.loc["Current Liabilities"]
        QuickRatio = ticker.info.get("quickRatio")
        return CurentRatio.iloc[0], QuickRatio

    def fuzzy_product(self, current_ratio, quick_ratio):
        current_ratio_raw = []
        quick_ratio_raw = []

        # ---------- Current Ratio fuzzification ----------
        for key, (a, b) in self.MAP_CURRENT_RATIO.items():
            if self.in_range(current_ratio, a, b):
                current_ratio_raw.append((key, a, b))

        CURRENT_RATIO_FUZZY = {}
        if len(current_ratio_raw) == 1:
            CURRENT_RATIO_FUZZY[current_ratio_raw[0][0]] = 1.0
        elif len(current_ratio_raw) == 2:
            (k1, a1, b1), (k2, a2, b2) = current_ratio_raw
            left = max(a1, a2)
            right = min(b1, b2)
            CURRENT_RATIO_FUZZY[k1] = self.linear_fuzzy(current_ratio, left, 1, right, 0)
            CURRENT_RATIO_FUZZY[k2] = self.linear_fuzzy(current_ratio, left, 0, right, 1)
        else:
            CURRENT_RATIO_FUZZY["UNKNOWN"] = 0.0

        # ---------- Quick Ratio fuzzification ----------
        for key, (a, b) in self.MAP_QUICK_RATIO.items():
            if self.in_range(quick_ratio, a, b):
                quick_ratio_raw.append((key, a, b))

        QUICK_RATIO_FUZZY = {}
        if len(quick_ratio_raw) == 1:
            QUICK_RATIO_FUZZY[quick_ratio_raw[0][0]] = 1.0
        elif len(quick_ratio_raw) == 2:
            (k1, a1, b1), (k2, a2, b2) = quick_ratio_raw
            left = max(a1, a2)
            right = min(b1, b2)
            QUICK_RATIO_FUZZY[k1] = self.linear_fuzzy(quick_ratio, left, 1, right, 0)
            QUICK_RATIO_FUZZY[k2] = self.linear_fuzzy(quick_ratio, left, 0, right, 1)
        else:
            QUICK_RATIO_FUZZY["UNKNOWN"] = 0.0

        return CURRENT_RATIO_FUZZY, QUICK_RATIO_FUZZY

    def Asset_Liquidity_Fuzzy(self, stock_id, get_level_label=False):
        current_ratio, quick_ratio = self.fetch_asset_liquidity(stock_id)
        current_ratio_fuzzy, quick_ratio_fuzzy = self.fuzzy_product(current_ratio, quick_ratio)
        combine_rules = super().infer_rules(current_ratio_fuzzy, quick_ratio_fuzzy, 'current_ratio', 'quick_ratio')
        final_score = self.defuzzify_sugeno(combine_rules)

        if get_level_label:
            level_label = self.map_fuzzy_output_centroid(final_score)
            return level_label, final_score
        return final_score

print
print("\n--- ROA Bottoming Trend Fuzzy System Results ---")
print("------------------------------------------------")
RULE_TABLE = {
    ("LOW", "DECLINING"): ("DETERIORATING", 25),
    ("LOW", "STABLE"): ("WEAK", 45),
    ("LOW", "IMPROVING"): ("NEUTRAL", 60),

    ("MEDIUM", "DECLINING"): ("WEAK", 45),
    ("MEDIUM", "STABLE"): ("NEUTRAL", 60),
    ("MEDIUM", "IMPROVING"): ("GOOD", 75),

    ("HIGH", "DECLINING"): ("NEUTRAL", 60),
    ("HIGH", "STABLE"): ("GOOD", 75),
    ("HIGH", "IMPROVING"): ("STRONG", 90),
}

MAP_ROA = {
    "LOW": (-np.inf, 0.06),
    "MEDIUM": (0.04, 0.08),
    "HIGH": (0.08, np.inf)
}

MAP_TREND = {
    "DECLINING": (-np.inf, 0.0),
    "STABLE": (-0.03, 0.06),
    "IMPROVING": (0.03, np.inf)
}

OUTPUT_LEVELS = {
    "DETERIORATING": 25,
    "WEAK": 45,
    "NEUTRAL": 60,
    "GOOD": 75,
    "STRONG": 90
}

fuzzy_system = ROABottomingTrendFuzzy(MAP_ROA, MAP_TREND, OUTPUT_LEVELS, RULE_TABLE)
for stock in STOCK_ID:
    try:
        level_label, final_score = fuzzy_system.ROA_Bottoming_Trend(stock, get_level_label=True)
        print(f"{stock}: Level - {level_label[0]}, Score - {final_score:.2f}")
    except Exception as e:
        print(f"{stock}: Error - {e}")


print("------------------------------------------------")
print("\n--- Asset Liquidity Fuzzy System Results ---")
print("------------------------------------------------")
RULE_TABLE = {
    # --- STRONG ---
    ("HIGH", "HIGH"): ("STRONG", 90),

    # --- GOOD ---
    ("MEDIUM", "HIGH"): ("GOOD", 75),
    ("HIGH", "MEDIUM"): ("GOOD", 75),

    # --- NEUTRAL ---
    ("LOW", "HIGH"): ("NEUTRAL", 60),
    ("MEDIUM", "MEDIUM"): ("NEUTRAL", 60),
    ("HIGH", "LOW"): ("NEUTRAL", 60),

    # --- WEAK ---
    ("LOW", "MEDIUM"): ("WEAK", 45),
    ("MEDIUM", "LOW"): ("WEAK", 45),

    # --- DETERIORATING ---
    ("LOW", "LOW"): ("DETERIORATING", 25),
}

MAP_CURRENT_RATIO = {
    "LOW":    (-np.inf, 1.0),   
    "MEDIUM": (0.8, 1.5),     
    "HIGH":   (1.3, np.inf)     
}

MAP_QUICK_RATIO = {
    "LOW":    (-np.inf, 1.0),   
    "MEDIUM": (0.8, 1.5),     
    "HIGH":   (1.3, np.inf)     
}

OUTPUT_LEVELS = {
    "DETERIORATING": 25,
    "WEAK": 45,
    "NEUTRAL": 60,
    "GOOD": 75,
    "STRONG": 90
}

fuzzy_system = AssetLiquidityFuzzy(MAP_CURRENT_RATIO, MAP_QUICK_RATIO, OUTPUT_LEVELS, RULE_TABLE)

for stock in STOCK_ID:
    try:
        level_label, final_score = fuzzy_system.Asset_Liquidity_Fuzzy(stock, get_level_label=True)
        print(f"{stock}: Level - {level_label[0]}, Score - {final_score:.2f}")
    except Exception as e:
        print(f"{stock}: Error - {e}")