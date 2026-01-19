import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product
import json
from datetime import datetime

# ==================== STEP TRACKER ====================
class StepTracker:
    """Track and display processing steps with rich visualizations"""
    def __init__(self, title="Processing Pipeline"):
        self.steps = []
        self.step_count = 0
        self.title = title
    
    def add_step(self, step_num, title, description="", details_dict=None, status="completed"):
        """Add a step with structured information"""
        self.steps.append({
            "number": step_num,
            "title": title,
            "description": description,
            "details": details_dict or {},
            "status": status,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    def display_header(self):
        """Display pipeline header"""
        st.markdown(f"## 🔄 {self.title}")
        st.markdown("---")
    
    def display_step(self, step):
        """Display a single step with rich formatting"""
        step_num = step["number"]
        title = step["title"]
        description = step["description"]
        details = step["details"]
        
        # Step header with progress indicator
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.markdown(f"### ✅ Step {step_num}: {title}")
            if description:
                st.markdown(f"*{description}*", help=None)
        with col2:
            st.markdown(f"<div style='text-align:right'><code>{step['timestamp']}</code></div>", unsafe_allow_html=True)
        
        # Display details if available
        if details:
            st.markdown("**Details:**")
            detail_cols = st.columns(len(details))
            for idx, (key, value) in enumerate(details.items()):
                with detail_cols[idx]:
                    st.metric(label=key, value=value)
        
        st.divider()
    
    def display_all(self):
        """Display all steps with progress bar"""
        self.display_header()
        
        # Progress bar
        progress_col, text_col = st.columns([0.8, 0.2])
        with progress_col:
            st.progress(1.0, "100% Complete")
        with text_col:
            st.metric("Total Steps", len(self.steps))
        
        st.markdown("")
        
        # Display each step
        for idx, step in enumerate(self.steps):
            # Step progress indicator
            step_progress = (idx + 1) / len(self.steps)
            st.markdown(f"**Progress: Step {idx + 1} of {len(self.steps)}**")
            
            self.display_step(step)


class DetailedPipeline:
    """Enhanced pipeline for detailed step-by-step analysis"""
    
    @staticmethod
    def display_configuration(title, config_dict):
        """Display configuration section"""
        st.markdown(f"#### 📋 {title}")
        config_df = pd.DataFrame([
            {"Setting": k, "Value": str(v)[:100]} 
            for k, v in config_dict.items()
        ])
        st.dataframe(config_df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def display_fuzzy_membership(title, fuzzy_dict):
        """Display fuzzy membership visualization"""
        st.markdown(f"#### 🎯 {title}")
        membership_data = []
        for key, value in fuzzy_dict.items():
            membership_data.append({
                "Category": key,
                "Membership": f"{value:.4f}",
                "Visual": "▓" * int(value * 20) + "░" * (20 - int(value * 20))
            })
        df = pd.DataFrame(membership_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def display_ranges(title, ranges_dict):
        """Display range configuration"""
        st.markdown(f"#### 📊 {title}")
        range_data = []
        for key, (min_val, max_val) in ranges_dict.items():
            min_display = f"{min_val:.4f}" if not np.isinf(min_val) else "-∞"
            max_display = f"{max_val:.4f}" if not np.isinf(max_val) else "+∞"
            range_data.append({
                "Category": key,
                "Min": min_display,
                "Max": max_display,
                "Range": f"[{min_display}, {max_display}]"
            })
        df = pd.DataFrame(range_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def display_rules(title, rules_list):
        """Display inference rules with results"""
        st.markdown(f"#### 🔗 {title}")
        rules_df = pd.DataFrame(rules_list)
        st.dataframe(rules_df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def display_raw_values(title, values_dict):
        """Display raw input values"""
        st.markdown(f"#### 📈 {title}")
        col1, col2 = st.columns(len(values_dict))
        for idx, (key, val) in enumerate(values_dict.items()):
            with st.columns(len(values_dict))[idx]:
                if isinstance(val, (int, float)):
                    st.metric(key, f"{val:.4f}")
                else:
                    st.metric(key, str(val)[:50])

# ==================== FUZZY BASE CLASS ====================
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
        """Generic rule inference combining two fuzzy membership dicts."""
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


# ==================== ROA BOTTOMING TREND FUZZY ====================
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
                if "Net Income" not in financials.index:
                    continue
                if "Total Assets" not in balance_sheet.index:
                    continue

                net_income = financials.loc["Net Income", col]
                total_assets = balance_sheet.loc["Total Assets", col]

                if pd.isna(net_income) or pd.isna(total_assets):
                    continue

                if total_assets == 0:
                    continue

                roa = net_income / total_assets

                if np.isnan(roa) or np.isinf(roa):
                    continue

                roa_by_year[col.year] = roa

            except Exception:
                continue

        if not roa_by_year:
            return None

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
            slope = self.slope_of_list(data)
        elif min_index == 0:
            slope = self.slope_of_list(data)
        else:
            slope = self.slope_of_list(data[min_index:])

        roe_finall_data = data[-1]
        roa_fuzzy, trend_fuzzy = self.fuzzy_product(roe_finall_data, slope)
        combine_rules = super().infer_rules(roa_fuzzy, trend_fuzzy, 'roa', 'trend')

        final_score = self.defuzzify_sugeno(combine_rules)
        
        if get_level_label:
            level_label = self.map_fuzzy_output_centroid(final_score)
            return level_label, final_score
        return final_score


# ==================== ASSET LIQUIDITY FUZZY ====================
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


# ==================== STREAMLIT APP ====================
st.set_page_config(page_title="Fuzzy Logic Stock Analyzer", layout="wide")
st.title("📊 Fuzzy Logic Stock Analyzer")

# Sidebar navigation
st.sidebar.title("Menu")
app_mode = st.sidebar.radio("Select Function:", ["ROA Bottoming Trend Fuzzy", "Asset Liquidity Fuzzy"])

# ==================== ROA BOTTOMING TREND TAB ====================
if app_mode == "ROA Bottoming Trend Fuzzy":
    st.header("🔍 ROA Bottoming Trend Fuzzy System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Enter Stock ID")
        stock_id = st.text_input("Stock ID (e.g., 5020.T):", value="5020.T")
    
    with col2:
        st.subheader("Configure OUTPUT_LEVELS")
        output_levels_input = st.text_area(
            "OUTPUT_LEVELS (JSON):",
            value='{"DETERIORATING": 25, "WEAK": 45, "NEUTRAL": 60, "GOOD": 75, "STRONG": 90}',
            height=100
        )
    
    st.subheader("Configure MAP_ROA")
    col_roa_1, col_roa_2, col_roa_3 = st.columns(3)
    with col_roa_1:
        roa_low_upper = st.number_input("ROA LOW - Upper bound:", value=0.05, step=0.01)
    with col_roa_2:
        roa_medium_lower = st.number_input("ROA MEDIUM - Lower bound:", value=0.03, step=0.01)
        roa_medium_upper = st.number_input("ROA MEDIUM - Upper bound:", value=0.10, step=0.01)
    with col_roa_3:
        roa_high_lower = st.number_input("ROA HIGH - Lower bound:", value=0.08, step=0.01)
    
    st.subheader("Configure MAP_TREND")
    col_trend_1, col_trend_2, col_trend_3 = st.columns(3)
    with col_trend_1:
        trend_declining_upper = st.number_input("TREND DECLINING - Upper bound:", value=-0.05, step=0.01)
    with col_trend_2:
        trend_stable_lower = st.number_input("TREND STABLE - Lower bound:", value=-0.07, step=0.01)
        trend_stable_upper = st.number_input("TREND STABLE - Upper bound:", value=0.05, step=0.01)
    with col_trend_3:
        trend_improving_lower = st.number_input("TREND IMPROVING - Lower bound:", value=0.03, step=0.01)
    
    st.subheader("Configure RULE_TABLE")
    rule_table_input = st.text_area(
        "RULE_TABLE (JSON):",
        value='''{
    "LOW-DECLINING": "DETERIORATING-25",
    "LOW-STABLE": "WEAK-45",
    "LOW-IMPROVING": "NEUTRAL-60",
    "MEDIUM-DECLINING": "WEAK-45",
    "MEDIUM-STABLE": "NEUTRAL-60",
    "MEDIUM-IMPROVING": "GOOD-75",
    "HIGH-DECLINING": "NEUTRAL-60",
    "HIGH-STABLE": "GOOD-75",
    "HIGH-IMPROVING": "STRONG-90"
}''',
        height=150
    )
    
    if st.button("🚀 Calculate ROA Bottoming Trend", key="roa_button"):
        try:
            pipeline = DetailedPipeline()
            tracker = StepTracker("ROA Bottoming Trend Fuzzy Analysis")
            tracker.display_header()
            
            # Step 1: Configuration Setup
            st.markdown("### 📋 Step 1: Configuration Setup")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Parsing Configurations...**")
                output_levels = json.loads(output_levels_input)
                pipeline.display_configuration("Output Level Mapping", output_levels)
            
            with col2:
                st.markdown("**Status:** ✅ Configuration Loaded")
                st.info(f"✓ Total output levels: {len(output_levels)}")
            
            tracker.add_step(1, "Configuration Setup", "Load and parse output levels", 
                           {"Output Levels": len(output_levels)})
            st.divider()
            
            # Step 2: Build Fuzzy Membership Maps
            st.markdown("### 📊 Step 2: Build Fuzzy Membership Maps")
            
            MAP_ROA = {
                "LOW": (-np.inf, np.nextafter(roa_low_upper, np.inf)),
                "MEDIUM": (roa_medium_lower, np.nextafter(roa_medium_upper, np.inf)),
                "HIGH": (roa_high_lower, np.inf)
            }
            
            MAP_TREND = {
                "DECLINING": (-np.inf, np.nextafter(trend_declining_upper, np.inf)),
                "STABLE": (trend_stable_lower, np.nextafter(trend_stable_upper, np.inf)),
                "IMPROVING": (trend_improving_lower, np.inf)
            }
            
            col1, col2 = st.columns(2)
            with col1:
                pipeline.display_ranges("ROA Fuzzy Membership Ranges", MAP_ROA)
            with col2:
                pipeline.display_ranges("Trend Fuzzy Membership Ranges", MAP_TREND)
            
            tracker.add_step(2, "Fuzzy Membership Maps", "Create fuzzy membership ranges",
                           {"ROA Categories": len(MAP_ROA), "Trend Categories": len(MAP_TREND)})
            st.divider()
            
            # Step 3: Parse Rule Table
            st.markdown("### 🔗 Step 3: Load Fuzzy Inference Rules")
            
            rule_table_dict = json.loads(rule_table_input)
            RULE_TABLE = {}
            for key, value in rule_table_dict.items():
                parts = key.split("-")
                roa_type, trend_type = parts[0], parts[1]
                label, score = value.split("-")
                RULE_TABLE[(roa_type, trend_type)] = (label, int(score))
            
            rules_display = []
            for key, value in rule_table_dict.items():
                rules_display.append({
                    "Rule": key,
                    "Decision": value.split("-")[0],
                    "Score": value.split("-")[1]
                })
            pipeline.display_rules("Fuzzy Inference Rules Table", rules_display)
            tracker.add_step(3, "Load Fuzzy Rules", "Parse and validate inference rules",
                           {"Total Rules": len(RULE_TABLE)})
            st.divider()
            
            # Step 4: Fetch Financial Data
            st.markdown("### 📈 Step 4: Fetch Stock Financial Data")
            
            with st.spinner(f"Fetching data for {stock_id}..."):
                fuzzy_system = ROABottomingTrendFuzzy(MAP_ROA, MAP_TREND, output_levels, RULE_TABLE)
                roa_data = fuzzy_system.fetch_roa_multi_year(stock_id)
            
            if roa_data:
                st.success(f"✅ Successfully retrieved {len(roa_data)} years of ROA data")
                
                # Display ROA history
                years_back = min(5, len(roa_data))
                st.markdown(f"**Last {years_back} years ROA values:**")
                roa_history = pd.DataFrame({
                    "Year Index": list(range(len(roa_data)-years_back, len(roa_data))),
                    "ROA Value": [f"{v:.4f}" for v in roa_data[-years_back:]]
                })
                st.dataframe(roa_history, use_container_width=True, hide_index=True)
                
                tracker.add_step(4, "Fetch Financial Data", f"Retrieved ROA history for {stock_id}",
                               {"Years Retrieved": len(roa_data), "Latest ROA": f"{roa_data[-1]:.4f}"})
            else:
                st.error(f"❌ No ROA data available for {stock_id}")
                raise ValueError(f"No data for {stock_id}")
            
            st.divider()
            
            # Step 5: Calculate ROA Trend Slope
            st.markdown("### 📐 Step 5: Calculate ROA Trend Slope")
            
            slope = fuzzy_system.slope_of_list(roa_data)
            roa_final = roa_data[-1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latest ROA", f"{roa_final:.4f}")
            with col2:
                st.metric("Slope (Trend)", f"{slope:.6f}")
            with col3:
                trend_direction = "📈 Improving" if slope > 0 else "📉 Declining"
                st.metric("Trend Direction", trend_direction)
            
            st.markdown(f"**Interpretation:** The ROA trend has a slope of {slope:.6f}, indicating " + 
                       ("an improving trend (positive slope)" if slope > 0 else "a declining trend (negative slope)"))
            
            tracker.add_step(5, "Calculate Trend Slope", "Compute linear regression slope",
                           {"Latest ROA": f"{roa_final:.4f}", "Slope": f"{slope:.6f}"})
            st.divider()
            
            # Step 6: Fuzzification
            st.markdown("### 🎯 Step 6: Fuzzification (Convert Crisp Values to Fuzzy)")
            
            roa_fuzzy, trend_fuzzy = fuzzy_system.fuzzy_product(roa_final, slope)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**ROA Fuzzy Membership:**")
                pipeline.display_fuzzy_membership("ROA Fuzzy Values", roa_fuzzy)
            with col2:
                st.markdown("**Trend Fuzzy Membership:**")
                pipeline.display_fuzzy_membership("Trend Fuzzy Values", trend_fuzzy)
            
            st.markdown(f"""
            **What is Fuzzification?**
            - Converting crisp (exact) values into fuzzy (uncertain) membership values
            - ROA of {roa_final:.4f} belongs to fuzzy categories with certain strengths
            - Slope of {slope:.6f} belongs to trend categories with certain strengths
            """)
            
            tracker.add_step(6, "Fuzzification", "Convert crisp to fuzzy membership values",
                           {"ROA Categories": len(roa_fuzzy), "Trend Categories": len(trend_fuzzy)})
            st.divider()
            
            # Step 7: Inference Engine
            st.markdown("### 🔗 Step 7: Inference Engine (Apply Fuzzy Rules)")
            
            combine_rules = fuzzy_system.infer_rules(roa_fuzzy, trend_fuzzy, 'roa', 'trend')
            
            # Display all fired rules
            st.markdown("**All Fired Rules and Their Strengths:**")
            rules_fired = []
            for rule in combine_rules:
                rules_fired.append({
                    "ROA": rule['roa'],
                    "Trend": rule['trend'],
                    "Fired Rule": rule['label'],
                    "Score": rule['score'],
                    "Firing Strength": f"{rule['weight']:.4f}",
                    "Contribution": f"{rule['weight'] * rule['score']:.4f}"
                })
            
            rules_df = pd.DataFrame(rules_fired)
            st.dataframe(rules_df, use_container_width=True, hide_index=True)
            
            st.markdown(f"**Total Rule Combinations Applied:** {len(combine_rules)}")
            
            tracker.add_step(7, "Inference Engine", "Apply all fuzzy rules",
                           {"Total Rules Fired": len(combine_rules)})
            st.divider()
            
            # Step 8: Defuzzification
            st.markdown("### 📊 Step 8: Defuzzification (Sugeno Method)")
            
            final_score = fuzzy_system.defuzzify_sugeno(combine_rules)
            
            st.markdown("**Sugeno Defuzzification Formula:**")
            st.markdown("```")
            st.markdown("Final Score = Σ(weight × score) / Σ(weight)")
            st.markdown("```")
            
            # Calculate components
            weighted_sum = sum(r['weight'] * r['score'] for r in combine_rules)
            weight_sum = sum(r['weight'] for r in combine_rules)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Weighted Sum", f"{weighted_sum:.4f}")
            with col2:
                st.metric("Weight Sum", f"{weight_sum:.4f}")
            with col3:
                st.metric("Final Score", f"{final_score:.2f}")
            
            tracker.add_step(8, "Defuzzification", "Convert fuzzy output to crisp value (Sugeno)",
                           {"Weighted Sum": f"{weighted_sum:.4f}", "Final Score": f"{final_score:.2f}"})
            st.divider()
            
            # Step 9: Decision Mapping
            st.markdown("### 🎯 Step 9: Map Score to Final Decision")
            
            level_label = fuzzy_system.map_fuzzy_output_centroid(final_score)
            
            # Display all output levels
            st.markdown("**Output Level Reference:**")
            level_data = []
            for label, target_score in output_levels.items():
                distance = abs(final_score - target_score)
                is_selected = "✅" if label == level_label[0] else ""
                level_data.append({
                    "Decision": is_selected,
                    "Level": label,
                    "Target Score": target_score,
                    "Distance from Actual": f"{distance:.2f}"
                })
            level_df = pd.DataFrame(level_data)
            st.dataframe(level_df, use_container_width=True, hide_index=True)
            
            tracker.add_step(9, "Decision Mapping", "Map score to final assessment level",
                           {"Selected Level": level_label[0], "Target Score": level_label[1]})
            st.divider()
            
            # Final Results
            st.markdown("---")
            st.markdown("## 🏆 Final Results")
            
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            with result_col1:
                st.metric("Stock ID", stock_id, delta=None)
            with result_col2:
                st.metric("Final Score", f"{final_score:.2f}")
            with result_col3:
                st.metric("Assessment Level", level_label[0])
            with result_col4:
                st.metric("Target Score", level_label[1])
            
            # Success message
            st.success("✅ Analysis Complete! All steps executed successfully.")
            
            tracker.add_step(10, "Analysis Complete", "All processing steps completed successfully",
                           {"Final Score": f"{final_score:.2f}", "Level": level_label[0]})
            
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Parsing Error: {e}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")




# ==================== ASSET LIQUIDITY TAB ====================
else:  # Asset Liquidity Fuzzy
    st.header("💧 Asset Liquidity Fuzzy System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Enter Stock ID")
        stock_id = st.text_input("Stock ID (e.g., 5020.T):", value="5020.T")
    
    with col2:
        st.subheader("Configure OUTPUT_LEVELS")
        output_levels_input = st.text_area(
            "OUTPUT_LEVELS (JSON):",
            value='{"DETERIORATING": 25, "WEAK": 45, "NEUTRAL": 60, "GOOD": 75, "STRONG": 90}',
            height=100
        )
    
    st.subheader("Configure MAP_CURRENT_RATIO")
    col_cr_1, col_cr_2, col_cr_3 = st.columns(3)
    with col_cr_1:
        cr_low_upper = st.number_input("CURRENT_RATIO LOW - Upper bound:", value=1.0, step=0.1)
    with col_cr_2:
        cr_medium_lower = st.number_input("CURRENT_RATIO MEDIUM - Lower bound:", value=0.8, step=0.1)
        cr_medium_upper = st.number_input("CURRENT_RATIO MEDIUM - Upper bound:", value=1.5, step=0.1)
    with col_cr_3:
        cr_high_lower = st.number_input("CURRENT_RATIO HIGH - Lower bound:", value=1.3, step=0.1)
    
    st.subheader("Configure MAP_QUICK_RATIO")
    col_qr_1, col_qr_2, col_qr_3 = st.columns(3)
    with col_qr_1:
        qr_low_upper = st.number_input("QUICK_RATIO LOW - Upper bound:", value=1.0, step=0.1)
    with col_qr_2:
        qr_medium_lower = st.number_input("QUICK_RATIO MEDIUM - Lower bound:", value=0.8, step=0.1)
        qr_medium_upper = st.number_input("QUICK_RATIO MEDIUM - Upper bound:", value=1.5, step=0.1)
    with col_qr_3:
        qr_high_lower = st.number_input("QUICK_RATIO HIGH - Lower bound:", value=1.3, step=0.1)
    
    st.subheader("Configure RULE_TABLE")
    rule_table_input = st.text_area(
        "RULE_TABLE (JSON):",
        value='''{
    "HIGH-HIGH": "STRONG-90",
    "MEDIUM-HIGH": "GOOD-75",
    "HIGH-MEDIUM": "GOOD-75",
    "LOW-HIGH": "NEUTRAL-60",
    "MEDIUM-MEDIUM": "NEUTRAL-60",
    "HIGH-LOW": "NEUTRAL-60",
    "LOW-MEDIUM": "WEAK-45",
    "MEDIUM-LOW": "WEAK-45",
    "LOW-LOW": "DETERIORATING-25"
}''',
        height=150
    )
    
    if st.button("🚀 Calculate Asset Liquidity", key="liquidity_button"):
        try:
            pipeline = DetailedPipeline()
            tracker = StepTracker("Asset Liquidity Fuzzy Analysis")
            tracker.display_header()
            
            # Step 1: Configuration Setup
            st.markdown("### 📋 Step 1: Configuration Setup")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Parsing Configurations...**")
                output_levels = json.loads(output_levels_input)
                pipeline.display_configuration("Output Level Mapping", output_levels)
            
            with col2:
                st.markdown("**Status:** ✅ Configuration Loaded")
                st.info(f"✓ Total output levels: {len(output_levels)}")
            
            tracker.add_step(1, "Configuration Setup", "Load and parse output levels", 
                           {"Output Levels": len(output_levels)})
            st.divider()
            
            # Step 2: Build Fuzzy Membership Maps
            st.markdown("### 📊 Step 2: Build Fuzzy Membership Maps")
            
            MAP_CURRENT_RATIO = {
                "LOW": (-np.inf, np.nextafter(cr_low_upper, np.inf)),
                "MEDIUM": (cr_medium_lower, np.nextafter(cr_medium_upper, np.inf)),
                "HIGH": (cr_high_lower, np.inf)
            }
            
            MAP_QUICK_RATIO = {
                "LOW": (-np.inf, np.nextafter(qr_low_upper, np.inf)),
                "MEDIUM": (qr_medium_lower, np.nextafter(qr_medium_upper, np.inf)),
                "HIGH": (qr_high_lower, np.inf)
            }
            
            col1, col2 = st.columns(2)
            with col1:
                pipeline.display_ranges("Current Ratio Fuzzy Membership Ranges", MAP_CURRENT_RATIO)
            with col2:
                pipeline.display_ranges("Quick Ratio Fuzzy Membership Ranges", MAP_QUICK_RATIO)
            
            tracker.add_step(2, "Fuzzy Membership Maps", "Create fuzzy membership ranges",
                           {"Current Ratio Categories": len(MAP_CURRENT_RATIO), "Quick Ratio Categories": len(MAP_QUICK_RATIO)})
            st.divider()
            
            # Step 3: Parse Rule Table
            st.markdown("### 🔗 Step 3: Load Fuzzy Inference Rules")
            
            rule_table_dict = json.loads(rule_table_input)
            RULE_TABLE = {}
            for key, value in rule_table_dict.items():
                parts = key.split("-")
                cr_type, qr_type = parts[0], parts[1]
                label, score = value.split("-")
                RULE_TABLE[(cr_type, qr_type)] = (label, int(score))
            
            rules_display = []
            for key, value in rule_table_dict.items():
                rules_display.append({
                    "Rule": key,
                    "Decision": value.split("-")[0],
                    "Score": value.split("-")[1]
                })
            pipeline.display_rules("Fuzzy Inference Rules Table", rules_display)
            tracker.add_step(3, "Load Fuzzy Rules", "Parse and validate inference rules",
                           {"Total Rules": len(RULE_TABLE)})
            st.divider()
            
            # Step 4: Fetch Financial Data
            st.markdown("### 📈 Step 4: Fetch Stock Financial Data")
            
            with st.spinner(f"Fetching balance sheet data for {stock_id}..."):
                fuzzy_system = AssetLiquidityFuzzy(MAP_CURRENT_RATIO, MAP_QUICK_RATIO, output_levels, RULE_TABLE)
                current_ratio, quick_ratio = fuzzy_system.fetch_asset_liquidity(stock_id)
            
            st.success(f"✅ Successfully retrieved balance sheet data")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Ratio", f"{current_ratio:.4f}", help="Current Assets / Current Liabilities")
            with col2:
                st.metric("Quick Ratio", f"{quick_ratio:.4f}", help="(Current Assets - Inventory) / Current Liabilities")
            
            tracker.add_step(4, "Fetch Financial Data", f"Retrieved balance sheet data for {stock_id}",
                           {"Current Ratio": f"{current_ratio:.4f}", "Quick Ratio": f"{quick_ratio:.4f}"})
            st.divider()
            
            # Step 5: Fuzzification
            st.markdown("### 🎯 Step 5: Fuzzification (Convert Crisp Values to Fuzzy)")
            
            current_ratio_fuzzy, quick_ratio_fuzzy = fuzzy_system.fuzzy_product(current_ratio, quick_ratio)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Current Ratio Fuzzy Membership:**")
                pipeline.display_fuzzy_membership("Current Ratio Fuzzy Values", current_ratio_fuzzy)
            with col2:
                st.markdown("**Quick Ratio Fuzzy Membership:**")
                pipeline.display_fuzzy_membership("Quick Ratio Fuzzy Values", quick_ratio_fuzzy)
            
            st.markdown(f"""
            **What is Fuzzification?**
            - Converting crisp (exact) values into fuzzy (uncertain) membership values
            - Current Ratio of {current_ratio:.4f} belongs to fuzzy categories with certain strengths
            - Quick Ratio of {quick_ratio:.4f} belongs to fuzzy categories with certain strengths
            """)
            
            tracker.add_step(5, "Fuzzification", "Convert crisp to fuzzy membership values",
                           {"Current Ratio Categories": len(current_ratio_fuzzy), "Quick Ratio Categories": len(quick_ratio_fuzzy)})
            st.divider()
            
            # Step 6: Inference Engine
            st.markdown("### 🔗 Step 6: Inference Engine (Apply Fuzzy Rules)")
            
            combine_rules = fuzzy_system.infer_rules(current_ratio_fuzzy, quick_ratio_fuzzy, 'current_ratio', 'quick_ratio')
            
            # Display all fired rules
            st.markdown("**All Fired Rules and Their Strengths:**")
            rules_fired = []
            for rule in combine_rules:
                rules_fired.append({
                    "Current Ratio": rule['current_ratio'],
                    "Quick Ratio": rule['quick_ratio'],
                    "Fired Rule": rule['label'],
                    "Score": rule['score'],
                    "Firing Strength": f"{rule['weight']:.4f}",
                    "Contribution": f"{rule['weight'] * rule['score']:.4f}"
                })
            
            rules_df = pd.DataFrame(rules_fired)
            st.dataframe(rules_df, use_container_width=True, hide_index=True)
            
            st.markdown(f"**Total Rule Combinations Applied:** {len(combine_rules)}")
            
            tracker.add_step(6, "Inference Engine", "Apply all fuzzy rules",
                           {"Total Rules Fired": len(combine_rules)})
            st.divider()
            
            # Step 7: Defuzzification
            st.markdown("### 📊 Step 7: Defuzzification (Sugeno Method)")
            
            final_score = fuzzy_system.defuzzify_sugeno(combine_rules)
            
            st.markdown("**Sugeno Defuzzification Formula:**")
            st.markdown("```")
            st.markdown("Final Score = Σ(weight × score) / Σ(weight)")
            st.markdown("```")
            
            # Calculate components
            weighted_sum = sum(r['weight'] * r['score'] for r in combine_rules)
            weight_sum = sum(r['weight'] for r in combine_rules)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Weighted Sum", f"{weighted_sum:.4f}")
            with col2:
                st.metric("Weight Sum", f"{weight_sum:.4f}")
            with col3:
                st.metric("Final Score", f"{final_score:.2f}")
            
            tracker.add_step(7, "Defuzzification", "Convert fuzzy output to crisp value (Sugeno)",
                           {"Weighted Sum": f"{weighted_sum:.4f}", "Final Score": f"{final_score:.2f}"})
            st.divider()
            
            # Step 8: Decision Mapping
            st.markdown("### 🎯 Step 8: Map Score to Final Decision")
            
            level_label = fuzzy_system.map_fuzzy_output_centroid(final_score)
            
            # Display all output levels
            st.markdown("**Output Level Reference:**")
            level_data = []
            for label, target_score in output_levels.items():
                distance = abs(final_score - target_score)
                is_selected = "✅" if label == level_label[0] else ""
                level_data.append({
                    "Decision": is_selected,
                    "Level": label,
                    "Target Score": target_score,
                    "Distance from Actual": f"{distance:.2f}"
                })
            level_df = pd.DataFrame(level_data)
            st.dataframe(level_df, use_container_width=True, hide_index=True)
            
            tracker.add_step(8, "Decision Mapping", "Map score to final assessment level",
                           {"Selected Level": level_label[0], "Target Score": level_label[1]})
            st.divider()
            
            # Final Results
            st.markdown("---")
            st.markdown("## 🏆 Final Results")
            
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            with result_col1:
                st.metric("Stock ID", stock_id, delta=None)
            with result_col2:
                st.metric("Final Score", f"{final_score:.2f}")
            with result_col3:
                st.metric("Assessment Level", level_label[0])
            with result_col4:
                st.metric("Target Score", level_label[1])
            
            # Success message
            st.success("✅ Analysis Complete! All steps executed successfully.")
            
            tracker.add_step(9, "Analysis Complete", "All processing steps completed successfully",
                           {"Final Score": f"{final_score:.2f}", "Level": level_label[0]})
            
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Parsing Error: {e}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")



st.sidebar.markdown("---")
st.sidebar.info("💡 Stock Analysis Application using Fuzzy Logic")
