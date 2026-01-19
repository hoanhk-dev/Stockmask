"""
Custom Streamlit Styling Module
Enhanced UI/UX for Fuzzy Logic Stock Analyzer
"""

def get_custom_css():
    """Returns custom CSS for enhanced UI"""
    return """
    <style>
    /* ==================== GLOBAL STYLING ==================== */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
        --warning-color: #ff9900;
        --dark-bg: #0f1419;
        --light-bg: #f8f9fa;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        color: #ffffff;
    }
    
    /* ==================== TYPOGRAPHY ==================== */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: 0.5px;
    }
    
    p {
        color: #e0e0e0;
        line-height: 1.6;
    }
    
    /* ==================== SIDEBAR ==================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1419 100%);
        border-right: 3px solid #1f77b4;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2 {
        color: #1f77b4;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 12px;
        margin-bottom: 15px;
    }
    
    [data-testid="stSidebarNav"] {
        padding: 20px 0;
    }
    
    /* ==================== CARDS & CONTAINERS ==================== */
    .card-container {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3a 100%);
        border: 1px solid #1f77b4;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .card-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 35px rgba(31, 119, 180, 0.4);
        border-color: #ff7f0e;
    }
    
    /* ==================== METRICS ==================== */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1f77b4 0%, #1a5fa0 100%);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(31, 119, 180, 0.3);
        border: 1px solid #2fa1de;
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        background: linear-gradient(135deg, #2fa1de 0%, #1f77b4 100%);
        box-shadow: 0 8px 25px rgba(31, 119, 180, 0.5);
        transform: translateY(-2px);
    }
    
    [data-testid="metric-container"] > div > div > span {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* ==================== BUTTONS ==================== */
    .stButton > button {
        background: linear-gradient(135deg, #1f77b4 0%, #2fa1de 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 14px 28px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2fa1de 0%, #1f77b4 100%);
        box-shadow: 0 8px 30px rgba(31, 119, 180, 0.5);
        transform: translateY(-3px);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* ==================== INPUTS & SELECTS ==================== */
    [data-testid="stSelectbox"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stTextArea"] label,
    [data-testid="stNumberInput"] label {
        color: #ff7f0e;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Selectbox dropdown button specifically */
    .stSelectbox > div > div:last-child {
        background-color: #1a1f2e !important;
        color: #ffffff !important;
        border: 2px solid #1f77b4 !important;
    }
    
    .stSelectbox > div > div:hover,
    .stTextInput > div > div > input:hover,
    .stTextArea textarea:hover,
    .stNumberInput > div > div > input:hover {
        border-color: #ff7f0e !important;
        box-shadow: 0 0 12px rgba(255, 127, 14, 0.2) !important;
    }
    
    .stSelectbox > div > div:focus,
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #2fa1de !important;
        box-shadow: 0 0 15px rgba(47, 161, 222, 0.3) !important;
    }
    
    /* ==================== TABS ==================== */
    [data-testid="stTabs"] [role="tab"] {
        background-color: #1a1f2e;
        color: #ffffff;
        border-radius: 8px 8px 0 0;
        border: 2px solid #1f77b4;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    [data-testid="stTabs"] [role="tab"]:hover {
        background-color: #252d3a;
        border-color: #ff7f0e;
    }
    
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1f77b4 0%, #2fa1de 100%);
        color: white;
        border-color: #ff7f0e;
    }
    
    /* ==================== EXPANDER ==================== */
    [data-testid="stExpander"] {
        background-color: #1a1f2e;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stExpander"]:hover {
        border-color: #ff7f0e;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.2);
    }
    
    [data-testid="stExpander"] label {
        color: #ff7f0e;
        font-weight: 600;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* ==================== DIVIDERS ==================== */
    .divider {
        border-top: 2px solid;
        border-image: linear-gradient(90deg, #1f77b4, #ff7f0e, #1f77b4) 1;
        margin: 25px 0;
    }
    
    hr {
        border-color: #1f77b4;
        opacity: 0.3;
    }
    
    /* ==================== DATAFRAMES ==================== */
    [data-testid="stDataFrame"] {
        background-color: #1a1f2e;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        overflow: hidden;
    }
    
    [data-testid="stDataFrame"] thead {
        background: linear-gradient(135deg, #1f77b4 0%, #2fa1de 100%) !important;
        color: white !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stDataFrame"] tbody tr:hover {
        background-color: rgba(31, 119, 180, 0.1) !important;
    }
    
    /* ==================== ALERTS & MESSAGES ==================== */
    .stSuccess {
        background: linear-gradient(135deg, rgba(44, 160, 44, 0.15) 0%, rgba(44, 160, 44, 0.05) 100%);
        border: 2px solid #2ca02c;
        border-radius: 10px;
        padding: 15px;
        color: #90ff90;
        font-weight: 600;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 153, 0, 0.15) 0%, rgba(255, 153, 0, 0.05) 100%);
        border: 2px solid #ff9900;
        border-radius: 10px;
        padding: 15px;
        color: #ffd580;
        font-weight: 600;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(214, 39, 40, 0.15) 0%, rgba(214, 39, 40, 0.05) 100%);
        border: 2px solid #d62728;
        border-radius: 10px;
        padding: 15px;
        color: #ff6b6b;
        font-weight: 600;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.15) 0%, rgba(31, 119, 180, 0.05) 100%);
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 15px;
        color: #80d4ff;
        font-weight: 600;
    }
    
    /* ==================== PROGRESS BAR ==================== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 50%, #1f77b4 100%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    /* ==================== STEPS & BADGES ==================== */
    .step-container {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3a 100%);
        border-left: 5px solid #ff7f0e;
        border-radius: 10px;
        padding: 18px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(255, 127, 14, 0.15);
        transition: all 0.3s ease;
    }
    
    .step-container:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 25px rgba(255, 127, 14, 0.3);
    }
    
    .step-number {
        display: inline-block;
        background: linear-gradient(135deg, #ff7f0e 0%, #ffb347 100%);
        color: #000;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        text-align: center;
        line-height: 40px;
        font-weight: 800;
        margin-right: 12px;
        font-size: 18px;
        box-shadow: 0 4px 10px rgba(255, 127, 14, 0.3);
    }
    
    /* ==================== ANIMATIONS ==================== */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(15px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: 200% 0;
        }
        100% {
            background-position: -200% 0;
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    .slide-in {
        animation: slideIn 0.6s ease-out;
    }
    
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* ==================== SCROLLBAR ==================== */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f1419;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2fa1de 0%, #ffb347 100%);
    }
    
    /* ==================== RESPONSIVE ==================== */
    @media (max-width: 768px) {
        h1, h2, h3 {
            font-size: 1.2em;
        }
        
        .stButton > button {
            padding: 12px 20px;
            font-size: 14px;
        }
    }
    </style>
    """

def apply_styles(st):
    """Apply custom CSS to Streamlit app"""
    st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header components
def render_main_header(title, subtitle=""):
    """Render main header with gradient background"""
    html = f"""
    <div style='background: linear-gradient(135deg, #1f77b4 0%, #ff7f0e 100%); 
                padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: center;
                box-shadow: 0 8px 25px rgba(31, 119, 180, 0.3);'>
        <h1 style='color: white; margin: 0; text-shadow: 3px 3px 8px rgba(0,0,0,0.4); font-size: 2.5em;'>
            {title}
        </h1>
    """
    if subtitle:
        html += f"<p style='color: #f0f0f0; margin: 10px 0 0 0; font-size: 1.1em;'>{subtitle}</p>"
    html += "</div>"
    return html

def render_section_header(title, icon=""):
    """Render section header with styling"""
    return f"""
    <div style='background: linear-gradient(135deg, #1a1f2e 0%, #252d3a 100%); 
                padding: 15px 20px; border-radius: 8px; margin-bottom: 15px;
                border-left: 5px solid #ff7f0e; box-shadow: 0 4px 10px rgba(31, 119, 180, 0.2);'>
        <p style='color: #ff7f0e; font-weight: 700; font-size: 1.1em; margin: 0; letter-spacing: 0.5px;'>
            {icon} {title.upper()}
        </p>
    </div>
    """

def render_metric_card(label, value, icon=""):
    """Render a styled metric card"""
    return f"""
    <div style='background: linear-gradient(135deg, #1f77b4 0%, #1a5fa0 100%);
                padding: 20px; border-radius: 10px; border: 1px solid #2fa1de;
                text-align: center; box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3);'>
        <p style='color: #b0d4f1; font-size: 0.9em; margin: 0 0 8px 0; text-transform: uppercase; 
                  font-weight: 600; letter-spacing: 0.5px;'>{icon} {label}</p>
        <h3 style='color: #ffffff; margin: 0; font-size: 1.8em;'>{value}</h3>
    </div>
    """

def render_success_box(message):
    """Render success message box"""
    return f"""
    <div style='background: linear-gradient(135deg, rgba(44, 160, 44, 0.2) 0%, rgba(44, 160, 44, 0.05) 100%);
                border: 2px solid #2ca02c; border-radius: 10px; padding: 18px;
                text-align: center; box-shadow: 0 4px 15px rgba(44, 160, 44, 0.2);'>
        <p style='color: #90ff90; font-weight: 700; margin: 0; font-size: 1.1em;'>✅ {message}</p>
    </div>
    """

def render_error_box(message):
    """Render error message box"""
    return f"""
    <div style='background: linear-gradient(135deg, rgba(214, 39, 40, 0.2) 0%, rgba(214, 39, 40, 0.05) 100%);
                border: 2px solid #d62728; border-radius: 10px; padding: 18px;
                text-align: center; box-shadow: 0 4px 15px rgba(214, 39, 40, 0.2);'>
        <p style='color: #ff6b6b; font-weight: 700; margin: 0; font-size: 1.1em;'>❌ {message}</p>
    </div>
    """
