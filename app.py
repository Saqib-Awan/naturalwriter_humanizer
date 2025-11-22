import streamlit as st
from humanizer import Humanizer
from detectors import AIDetector
from text_analysis import TextAnalyzer

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TrueVoice AI",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* 1. Main Background */
    .stApp {
        background: linear-gradient(120deg, #fdfbf7 0%, #e6f7efff 50%, #e0f2fe 100%);
        background-attachment: fixed;
    }

    /* 2. Remove Defaults */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* 3. Text Areas (Paper Look) */
    .stTextArea textarea {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 15px;
        font-size: 16px;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #333;
        min-height: 400px; /* Fixed height for consistency */
    }
    .stTextArea textarea:focus {
        border-color: #4da6ff;
        box-shadow: 0 0 0 2px rgba(77, 166, 255, 0.2);
    }

    /* 4. The Action Buttons (Both Shared Style) */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4da6ff 0%, #0077cc 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 50px;
        font-size: 16px; /* Slightly smaller text to fit side-by-side */
        transition: all 0.3s ease;
        margin-top: 5px; 
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 119, 204, 0.3);
    }

    /* 5. Header */
    .logo-text {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        font-size: 32px;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 5px;
    }
    .tagline {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 16px;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 25px;
    }
    
    /* 6. Stats Cards */
    .metric-card {
        background: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* 7. Word Count Styling */
    .word-count {
        font-size: 14px;
        color: #888;
        margin-top: -10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
@st.cache_resource
def get_humanizer():
    return Humanizer()

@st.cache_resource
def get_detector():
    return AIDetector()

@st.cache_resource
def get_analyzer():
    return TextAnalyzer()

humanizer_bot = get_humanizer()
detector_bot = get_detector()
analyzer_bot = get_analyzer()

# --- UI HEADER ---
st.markdown('<div class="logo-text">✒️ TrueVoice AI</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">The Free, Powerful AI-to-Human Converter</div>', unsafe_allow_html=True)

# --- CONTROLS (Top Bar) ---
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    mode = st.selectbox(
        "Select Humanization Strength", 
        ["Standard (Balanced)", "Ghost (Aggressive - Best Bypass)", "Lite (Fast)"],
        index=1 
    )

# --- MAIN EDITOR AREA ---
col1, col_center, col2 = st.columns([1, 0.05, 1]) # Reduced spacer width

# --- LEFT COLUMN: INPUT + BUTTONS ---
with col1:
    st.markdown("### Input (AI Text)")
    input_text = st.text_area("Input", height=400, label_visibility="collapsed", placeholder="Paste your AI-generated text here...")
    
    # Word count immediately below box
    word_count = len(input_text.split()) if input_text else 0
    if word_count > 500:
        st.markdown(f"<div class='word-count' style='color:red;'>⚠️ {word_count}/500 Words</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='word-count'>{word_count}/500 Words</div>", unsafe_allow_html=True)

    # --- BUTTONS (Inside Left Column) ---
    # Create two tight columns for the buttons
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        if st.button("Check for AI"):
            if not input_text:
                st.warning("Paste text first.")
            else:
                with st.spinner("Scanning..."):
                    scan_result = detector_bot.detect(input_text)
                    if scan_result['score'] > 50:
                        st.toast(f"⚠️ AI Detected! Probability: {scan_result['score']}%", icon="🤖")
                    else:
                        st.toast(f"✅ Looks Human! AI Score: {scan_result['score']}%", icon="👤")
    
    with btn_col2:
        if st.button("Humanize Text"):
            if not input_text:
                st.warning("Please enter some text first.")
            elif word_count > 500:
                st.error("Please reduce your text to under 500 words.")
            else:
                with st.spinner("Injecting human nuances... (Approx 10s)"):
                    try:
                        output = humanizer_bot.humanize(input_text, level=mode)
                        st.session_state['result'] = output
                        st.rerun() 
                    except Exception as e:
                        st.error(f"Error: {e}")

# --- RIGHT COLUMN: OUTPUT ---
with col2:
    st.markdown("### Humanized Text")
    result_text = st.session_state.get('result', "")
    st.text_area("Output", value=result_text, height=400, label_visibility="collapsed", placeholder="Your humanized text will appear here...")
    
    # Word count immediately below box
    if result_text:
        stats = analyzer_bot.analyze(result_text)
        st.markdown(f"<div class='word-count'>{stats.get('word_count', 0)} Words</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='word-count'>0 Words</div>", unsafe_allow_html=True)

# --- ANALYTICS FOOTER ---
if st.session_state.get('result'):
    st.markdown("<br>", unsafe_allow_html=True)
    scan = detector_bot.detect(st.session_state['result'])
    
    k1, k2, k3 = st.columns(3)
    
    score_color = "#4CAF50" 
    if scan['score'] > 40: score_color = "#FFC107" 
    if scan['score'] > 70: score_color = "#FF4B4B" 
    
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:14px; color:#888;">Detection Score</div>
            <div style="font-size:24px; font-weight:bold; color:{score_color};">{scan['score']}% AI</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:14px; color:#888;">Safety Level</div>
            <div style="font-size:24px; font-weight:bold; color:#333;">{"Safe ✅" if scan['score'] < 30 else "Risky ⚠️"}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:14px; color:#888;">Burstiness</div>
            <div style="font-size:24px; font-weight:bold; color:#333;">High ⚡</div>
        </div>
        """, unsafe_allow_html=True)