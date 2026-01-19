import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product
import json
from datetime import datetime

# Initialize session state for custom inputs
if "roa_custom_stock" not in st.session_state:
    st.session_state.roa_custom_stock = None
if "asset_custom_stock" not in st.session_state:
    st.session_state.asset_custom_stock = None

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

# Predefined Stock IDs
STOCK_ID_LIST = [
    "5020.T", "8058.T", "6920.T", "8001.T", "6952.T",
    "4393.T", "9104.T", "7974.T", "7203.T", "2432.T",
    "4689.T", "4063.T", "5108.T", "9697.T", "9983.T",
    "9984.T", "4568.T", "1812.T", "7011.T", "8801.T",
    "3635.T", "4816.T", "8306.T", "5411.T", "6857.T"
]

# Sidebar navigation
st.sidebar.title("Menu")
app_mode = st.sidebar.radio("Select Function:", ["ROA Bottoming Trend Fuzzy", "Asset Liquidity Fuzzy"])

# ==================== ROA BOTTOMING TREND TAB ====================
if app_mode == "ROA Bottoming Trend Fuzzy":
    st.header("🔍 ROA Bottoming Trend Fuzzy System")
    
    # Input Stock ID with dropdown
    st.subheader("📊 Input Stock ID")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_id = st.selectbox(
            "Select Stock ID:",
            [None] + STOCK_ID_LIST,
            format_func=lambda x: "Select Stock ID..." if x is None else x,
            key="roa_dropdown"
        )
    
    with col2:
        custom_input = st.text_input(
            "Or type custom:",
            value="",
            key="roa_custom",
            placeholder="e.g., AAPL"
        )
    
    # Use custom input if provided, otherwise use dropdown selection
    if custom_input.strip():
        stock_id = custom_input.strip()
    elif stock_id is None:
        st.warning("⚠️ Please select or enter a Stock ID")
        st.stop()
    
    # ========== Configuration Expander ==========
    with st.expander("⚙️ Configuration Settings", expanded=False):
        st.markdown("### Output Levels")
        output_levels_input = st.text_area(
            "OUTPUT_LEVELS (JSON):",
            value='{"DETERIORATING": 25, "WEAK": 45, "NEUTRAL": 60, "GOOD": 75, "STRONG": 90}',
            height=80,
            key="output_levels_roa"
        )
        
        st.markdown("### ROA Fuzzy Membership Ranges")
        col_roa_1, col_roa_2, col_roa_3 = st.columns(3)
        with col_roa_1:
            roa_low_upper = st.number_input("ROA LOW - Upper bound:", value=0.05, step=0.01, key="roa_low")
        with col_roa_2:
            roa_medium_lower = st.number_input("ROA MEDIUM - Lower bound:", value=0.03, step=0.01, key="roa_med_low")
            roa_medium_upper = st.number_input("ROA MEDIUM - Upper bound:", value=0.10, step=0.01, key="roa_med_up")
        with col_roa_3:
            roa_high_lower = st.number_input("ROA HIGH - Lower bound:", value=0.08, step=0.01, key="roa_high")
        
        st.markdown("### Trend Fuzzy Membership Ranges")
        col_trend_1, col_trend_2, col_trend_3 = st.columns(3)
        with col_trend_1:
            trend_declining_upper = st.number_input("TREND DECLINING - Upper bound:", value=-0.05, step=0.01, key="trend_dec")
        with col_trend_2:
            trend_stable_lower = st.number_input("TREND STABLE - Lower bound:", value=-0.07, step=0.01, key="trend_sta_low")
            trend_stable_upper = st.number_input("TREND STABLE - Upper bound:", value=0.05, step=0.01, key="trend_sta_up")
        with col_trend_3:
            trend_improving_lower = st.number_input("TREND IMPROVING - Lower bound:", value=0.03, step=0.01, key="trend_imp")
        
        st.markdown("### Fuzzy Inference Rules")
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
            height=300,
            key="rule_table_roa"
        )
    
    st.divider()
    
    # ========== Calculate Button ==========
    if st.button("🚀 Calculate ROA Bottoming Trend", key="roa_button_main", use_container_width=True):
        try:
            pipeline = DetailedPipeline()
            tracker = StepTracker("ROA Bottoming Trend Fuzzy Analysis")
            tracker.display_header()
            
            # Step 1: Membership Function Configuration
            st.markdown("### 1️⃣ Membership Function Configuration")
            
            output_levels = json.loads(output_levels_input)
            
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
                st.markdown("**ROA Membership Ranges:**")
                pipeline.display_ranges("ROA Categories", MAP_ROA)
            with col2:
                st.markdown("**Trend Membership Ranges:**")
                pipeline.display_ranges("Trend Categories", MAP_TREND)
            
            tracker.add_step(1, "Membership Function Configuration", "Define fuzzy membership ranges",
                           {"ROA Categories": len(MAP_ROA), "Trend Categories": len(MAP_TREND)})
            st.divider()
            
            # Step 2: Calculate
            st.markdown("### 2️⃣ Calculate & Fetch Data")
            
            with st.spinner(f"Fetching data for {stock_id}..."):
                rule_table_dict = json.loads(rule_table_input)
                RULE_TABLE = {}
                for key, value in rule_table_dict.items():
                    parts = key.split("-")
                    roa_type, trend_type = parts[0], parts[1]
                    label, score = value.split("-")
                    RULE_TABLE[(roa_type, trend_type)] = (label, int(score))
                
                fuzzy_system = ROABottomingTrendFuzzy(MAP_ROA, MAP_TREND, output_levels, RULE_TABLE)
                roa_data = fuzzy_system.fetch_roa_multi_year(stock_id)
            
            if not roa_data:
                st.error(f"❌ No ROA data available for {stock_id}")
                st.stop()
            
            st.success(f"✅ Retrieved {len(roa_data)} years of ROA data")
            
            # Calculate slope
            slope = fuzzy_system.slope_of_list(roa_data)
            roa_final = roa_data[-1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Latest ROA", f"{roa_final:.4f}")
            with col2:
                st.metric("Slope", f"{slope:.6f}")
            with col3:
                trend_dir = "📈 Improving" if slope > 0 else "📉 Declining"
                st.metric("Trend", trend_dir)
            
            tracker.add_step(2, "Calculate & Fetch Data", f"Retrieved data and calculated trend",
                           {"Years": len(roa_data), "Latest ROA": f"{roa_final:.4f}", "Slope": f"{slope:.6f}"})
            st.divider()
            
            # Step 3: Fuzzification
            st.markdown("### 3️⃣ Fuzzification")
            
            roa_fuzzy, trend_fuzzy = fuzzy_system.fuzzy_product(roa_final, slope)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**ROA Membership Values:**")
                pipeline.display_fuzzy_membership("ROA Fuzzy", roa_fuzzy)
            with col2:
                st.markdown("**Trend Membership Values:**")
                pipeline.display_fuzzy_membership("Trend Fuzzy", trend_fuzzy)
            
            tracker.add_step(3, "Fuzzification", "Convert crisp values to fuzzy memberships",
                           {"ROA Fuzzy": len(roa_fuzzy), "Trend Fuzzy": len(trend_fuzzy)})
            st.divider()
            
            # Step 4: Rules
            st.markdown("### 4️⃣ Fuzzy Inference Rules")
            
            combine_rules = fuzzy_system.infer_rules(roa_fuzzy, trend_fuzzy, 'roa', 'trend')
            
            rules_fired = []
            for rule in combine_rules:
                rules_fired.append({
                    "ROA": rule['roa'],
                    "Trend": rule['trend'],
                    "Rule": rule['label'],
                    "Score": rule['score'],
                    "Weight": f"{rule['weight']:.4f}",
                    "Contribution": f"{rule['weight'] * rule['score']:.4f}"
                })
            
            st.dataframe(pd.DataFrame(rules_fired), use_container_width=True, hide_index=True)
            st.markdown(f"**Total Rule Combinations:** {len(combine_rules)}")
            
            tracker.add_step(4, "Fuzzy Inference Rules", "Apply all fuzzy rules",
                           {"Rules Fired": len(combine_rules)})
            st.divider()
            
            # Step 5: Defuzzification
            st.markdown("### 5️⃣ Defuzzification (Sugeno Method)")
            
            final_score = fuzzy_system.defuzzify_sugeno(combine_rules)
            
            st.markdown("**Formula:** Final Score = Σ(weight × score) / Σ(weight)")
            
            weighted_sum = sum(r['weight'] * r['score'] for r in combine_rules)
            weight_sum = sum(r['weight'] for r in combine_rules)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Weighted Sum", f"{weighted_sum:.4f}")
            with col2:
                st.metric("Weight Sum", f"{weight_sum:.4f}")
            with col3:
                st.metric("Final Score", f"{final_score:.2f}")
            
            tracker.add_step(5, "Defuzzification", "Convert fuzzy output to crisp score",
                           {"Final Score": f"{final_score:.2f}"})
            st.divider()
            
            # Step 6: Final Result
            st.markdown("### 6️⃣ Final Result")
            
            level_label = fuzzy_system.map_fuzzy_output_centroid(final_score)
            
            level_data = []
            for label, target_score in output_levels.items():
                distance = abs(final_score - target_score)
                is_selected = "✅" if label == level_label[0] else ""
                level_data.append({
                    "": is_selected,
                    "Level": label,
                    "Target": target_score,
                    "Distance": f"{distance:.2f}"
                })
            
            st.dataframe(pd.DataFrame(level_data), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Stock", stock_id)
            with col2:
                st.metric("Score", f"{final_score:.2f}")
            with col3:
                st.metric("Level", level_label[0])
            with col4:
                st.metric("Target", level_label[1])
            
            st.success("✅ Analysis Complete!")
            
            tracker.add_step(6, "Final Result", "Assessment completed",
                           {"Level": level_label[0], "Score": f"{final_score:.2f}"})
            
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Error: {e}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")





# ==================== ASSET LIQUIDITY TAB ====================
else:  # Asset Liquidity Fuzzy
    st.header("💧 Asset Liquidity Fuzzy System")
    
    # Input Stock ID with dropdown
    st.subheader("📊 Input Stock ID")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        stock_id = st.selectbox(
            "Select Stock ID:",
            [None] + STOCK_ID_LIST,
            format_func=lambda x: "Select Stock ID..." if x is None else x,
            key="asset_dropdown"
        )
    
    with col2:
        custom_input = st.text_input(
            "Or type custom:",
            value="",
            key="asset_custom",
            placeholder="e.g., AAPL"
        )
    
    # Use custom input if provided, otherwise use dropdown selection
    if custom_input.strip():
        stock_id = custom_input.strip()
    elif stock_id is None:
        st.warning("⚠️ Please select or enter a Stock ID")
        st.stop()
    
    # ========== Configuration Expander ==========
    with st.expander("⚙️ Configuration Settings", expanded=False):
        st.markdown("### Output Levels")
        output_levels_input = st.text_area(
            "Output Level Mapping (JSON):",
            value='{"DETERIORATING": 25, "WEAK": 45, "NEUTRAL": 60, "GOOD": 75, "STRONG": 90}',
            height=100,
            key="output_levels_asset"
        )
        
        st.markdown("### Current Ratio Membership Ranges")
        col_cr_1, col_cr_2, col_cr_3 = st.columns(3)
        with col_cr_1:
            cr_low_upper = st.number_input("CR LOW - Upper:", value=1.0, step=0.1, key="cr_low_asset")
        with col_cr_2:
            cr_medium_lower = st.number_input("CR MED - Lower:", value=0.8, step=0.1, key="cr_med_low_asset")
            cr_medium_upper = st.number_input("CR MED - Upper:", value=1.5, step=0.1, key="cr_med_up_asset")
        with col_cr_3:
            cr_high_lower = st.number_input("CR HIGH - Lower:", value=1.3, step=0.1, key="cr_high_asset")
        
        st.markdown("### Quick Ratio Membership Ranges")
        col_qr_1, col_qr_2, col_qr_3 = st.columns(3)
        with col_qr_1:
            qr_low_upper = st.number_input("QR LOW - Upper:", value=1.0, step=0.1, key="qr_low_asset")
        with col_qr_2:
            qr_medium_lower = st.number_input("QR MED - Lower:", value=0.8, step=0.1, key="qr_med_low_asset")
            qr_medium_upper = st.number_input("QR MED - Upper:", value=1.5, step=0.1, key="qr_med_up_asset")
        with col_qr_3:
            qr_high_lower = st.number_input("QR HIGH - Lower:", value=1.3, step=0.1, key="qr_high_asset")
        
        st.markdown("### Fuzzy Inference Rules")
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
            height=300,
            key="rule_table_asset"
        )
    
    # ========== Calculate Button ==========
    if st.button("🚀 Calculate Asset Liquidity", key="liquidity_button_main", use_container_width=True):
        try:
            pipeline = DetailedPipeline()
            tracker = StepTracker("Asset Liquidity Fuzzy Analysis")
            tracker.display_header()
            
            # Step 1: Membership Function Configuration
            st.markdown("### 1️⃣ Membership Function Configuration")
            
            output_levels = json.loads(output_levels_input)
            
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
                st.markdown("**Current Ratio Membership Ranges:**")
                pipeline.display_ranges("Current Ratio Categories", MAP_CURRENT_RATIO)
            with col2:
                st.markdown("**Quick Ratio Membership Ranges:**")
                pipeline.display_ranges("Quick Ratio Categories", MAP_QUICK_RATIO)
            
            tracker.add_step(1, "Membership Function Configuration", "Define fuzzy membership ranges",
                           {"Current Ratio Categories": len(MAP_CURRENT_RATIO), "Quick Ratio Categories": len(MAP_QUICK_RATIO)})
            st.divider()
            
            # Step 2: Calculate
            st.markdown("### 2️⃣ Calculate & Fetch Data")
            
            with st.spinner(f"Fetching data for {stock_id}..."):
                rule_table_dict = json.loads(rule_table_input)
                RULE_TABLE = {}
                for key, value in rule_table_dict.items():
                    parts = key.split("-")
                    cr_type, qr_type = parts[0], parts[1]
                    label, score = value.split("-")
                    RULE_TABLE[(cr_type, qr_type)] = (label, int(score))
                
                fuzzy_system = AssetLiquidityFuzzy(MAP_CURRENT_RATIO, MAP_QUICK_RATIO, output_levels, RULE_TABLE)
                current_ratio, quick_ratio = fuzzy_system.fetch_asset_liquidity(stock_id)
            
            st.success(f"✅ Retrieved balance sheet data for {stock_id}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Ratio", f"{current_ratio:.4f}")
            with col2:
                st.metric("Quick Ratio", f"{quick_ratio:.4f}")
            
            tracker.add_step(2, "Calculate & Fetch Data", f"Retrieved data for {stock_id}",
                           {"Current Ratio": f"{current_ratio:.4f}", "Quick Ratio": f"{quick_ratio:.4f}"})
            st.divider()
            
            # Step 3: Fuzzification
            st.markdown("### 3️⃣ Fuzzification")
            
            current_ratio_fuzzy, quick_ratio_fuzzy = fuzzy_system.fuzzy_product(current_ratio, quick_ratio)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Current Ratio Membership Values:**")
                pipeline.display_fuzzy_membership("Current Ratio Fuzzy", current_ratio_fuzzy)
            with col2:
                st.markdown("**Quick Ratio Membership Values:**")
                pipeline.display_fuzzy_membership("Quick Ratio Fuzzy", quick_ratio_fuzzy)
            
            tracker.add_step(3, "Fuzzification", "Convert crisp values to fuzzy memberships",
                           {"Current Ratio Fuzzy": len(current_ratio_fuzzy), "Quick Ratio Fuzzy": len(quick_ratio_fuzzy)})
            st.divider()
            
            # Step 4: Rules
            st.markdown("### 4️⃣ Fuzzy Inference Rules")
            
            combine_rules = fuzzy_system.infer_rules(current_ratio_fuzzy, quick_ratio_fuzzy, 'current_ratio', 'quick_ratio')
            
            rules_fired = []
            for rule in combine_rules:
                rules_fired.append({
                    "Current Ratio": rule['current_ratio'],
                    "Quick Ratio": rule['quick_ratio'],
                    "Rule": rule['label'],
                    "Score": rule['score'],
                    "Weight": f"{rule['weight']:.4f}",
                    "Contribution": f"{rule['weight'] * rule['score']:.4f}"
                })
            
            st.dataframe(pd.DataFrame(rules_fired), use_container_width=True, hide_index=True)
            st.markdown(f"**Total Rule Combinations:** {len(combine_rules)}")
            
            tracker.add_step(4, "Fuzzy Inference Rules", "Apply all fuzzy rules",
                           {"Rules Fired": len(combine_rules)})
            st.divider()
            
            # Step 5: Defuzzification
            st.markdown("### 5️⃣ Defuzzification (Sugeno Method)")
            
            final_score = fuzzy_system.defuzzify_sugeno(combine_rules)
            
            st.markdown("**Formula:** Final Score = Σ(weight × score) / Σ(weight)")
            
            weighted_sum = sum(r['weight'] * r['score'] for r in combine_rules)
            weight_sum = sum(r['weight'] for r in combine_rules)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Weighted Sum", f"{weighted_sum:.4f}")
            with col2:
                st.metric("Weight Sum", f"{weight_sum:.4f}")
            with col3:
                st.metric("Final Score", f"{final_score:.2f}")
            
            tracker.add_step(5, "Defuzzification", "Convert fuzzy output to crisp score",
                           {"Final Score": f"{final_score:.2f}"})
            st.divider()
            
            # Step 6: Final Result
            st.markdown("### 6️⃣ Final Result")
            
            level_label = fuzzy_system.map_fuzzy_output_centroid(final_score)
            
            level_data = []
            for label, target_score in output_levels.items():
                distance = abs(final_score - target_score)
                is_selected = "✅" if label == level_label[0] else ""
                level_data.append({
                    "": is_selected,
                    "Level": label,
                    "Target": target_score,
                    "Distance": f"{distance:.2f}"
                })
            
            st.dataframe(pd.DataFrame(level_data), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Stock", stock_id)
            with col2:
                st.metric("Score", f"{final_score:.2f}")
            with col3:
                st.metric("Level", level_label[0])
            with col4:
                st.metric("Target", level_label[1])
            
            st.success("✅ Analysis Complete!")
            
            tracker.add_step(6, "Final Result", "Assessment completed",
                           {"Level": level_label[0], "Score": f"{final_score:.2f}"})
            
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Error: {e}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            
            tracker.add_step(9, "Analysis Complete", "All processing steps completed successfully",
                           {"Final Score": f"{final_score:.2f}", "Level": level_label[0]})
            
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Parsing Error: {e}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")



st.sidebar.markdown("---")
st.sidebar.info("💡 Stock Analysis Application using Fuzzy Logic")
