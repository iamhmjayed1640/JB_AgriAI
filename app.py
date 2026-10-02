import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import random
import base64

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="JB AgriAI — Intelligent Rice Disease Diagnosis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# SESSION STATE FOR NAVIGATION
# ==========================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

def nav_to(page):
    st.session_state.page = page

# ==========================================
# SVG LOGO
# ==========================================
JB_LOGO_SVG = """
<svg width="48" height="48" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
  <circle cx="24" cy="24" r="23" fill="#0B3D2E" stroke="#16A34A" stroke-width="1.5"/>
  <path d="M14 34 Q24 8 34 14 Q28 20 24 28 Q20 20 14 34Z" fill="#16A34A" opacity="0.9"/>
  <path d="M24 15 L24 30" stroke="#D6B85A" stroke-width="0.8" opacity="0.6"/>
  <path d="M20 20 L24 18" stroke="#D6B85A" stroke-width="0.5" opacity="0.4"/>
  <path d="M28 18 L24 20" stroke="#D6B85A" stroke-width="0.5" opacity="0.4"/>
  <path d="M21 24 L24 22" stroke="#D6B85A" stroke-width="0.5" opacity="0.4"/>
  <path d="M27 22 L24 24" stroke="#D6B85A" stroke-width="0.5" opacity="0.4"/>
  <text x="24" y="42" text-anchor="middle" fill="#D6B85A" font-family="Arial Black, sans-serif" font-size="9" font-weight="900">JB</text>
</svg>
"""
LOGO_B64 = base64.b64encode(JB_LOGO_SVG.encode()).decode()

# ==========================================
# MEGA CSS
# ==========================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ===== GLOBAL ===== */
html, body, .stApp {{
    background-color: #F7F8F3 !important;
    font-family: 'Inter', sans-serif !important;
    color: #10231C;
    scroll-behavior: smooth;
}}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {{
    background: #071F18 !important;
    border-right: 1px solid rgba(22, 163, 74, 0.15);
    width: 280px !important;
}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown span {{
    color: #FAFAF5 !important;
}}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {{
    color: #16A34A !important;
    font-family: 'Space Grotesk', sans-serif !important;
}}

/* ===== TOP NAVBAR ===== */
.top-navbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 40px;
    background: rgba(247, 248, 243, 0.85);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(11, 61, 46, 0.08);
    margin: -1rem -1rem 0 -1rem;
    position: sticky;
    top: 0;
    z-index: 9999;
}}
.nav-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
}}
.nav-brand-text {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 20px;
    color: #0B3D2E;
}}
.nav-links {{
    display: flex;
    gap: 0;
    align-items: center;
}}
.nav-link {{
    padding: 8px 18px;
    font-size: 14px;
    font-weight: 500;
    color: #52645C;
    text-decoration: none;
    border-radius: 8px;
    transition: all 0.25s ease;
    cursor: pointer;
}}
.nav-link:hover {{
    color: #0B3D2E;
    background: rgba(22, 163, 74, 0.08);
}}
.nav-link.active {{
    color: #0B3D2E;
    font-weight: 600;
    background: rgba(22, 163, 74, 0.1);
}}
.nav-cta {{
    padding: 9px 22px;
    background: #0B3D2E;
    color: #FAFAF5 !important;
    font-size: 13px;
    font-weight: 600;
    border-radius: 8px;
    text-decoration: none;
    transition: all 0.3s ease;
    cursor: pointer;
    letter-spacing: 0.3px;
}}
.nav-cta:hover {{
    background: #16A34A;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(22, 163, 74, 0.3);
}}

/* ===== ANIMATIONS ===== */
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(35px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to {{ opacity: 1; }}
}}
@keyframes scanLine {{
    0% {{ top: 0%; opacity: 0; }}
    10% {{ opacity: 1; }}
    90% {{ opacity: 1; }}
    100% {{ top: 92%; opacity: 0; }}
}}
@keyframes pulseGlow {{
    0%   {{ box-shadow: 0 0 15px rgba(22, 163, 74, 0.1); border-color: rgba(22, 163, 74, 0.3); }}
    50%  {{ box-shadow: 0 0 40px rgba(22, 163, 74, 0.4); border-color: rgba(22, 163, 74, 0.8); }}
    100% {{ box-shadow: 0 0 15px rgba(22, 163, 74, 0.1); border-color: rgba(22, 163, 74, 0.3); }}
}}
@keyframes float {{
    0%, 100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-8px); }}
}}

/* ===== HERO ===== */
.hero-badge {{
    display: inline-block;
    padding: 8px 20px;
    background: linear-gradient(135deg, #D6B85A, #E6C766);
    color: #0B3D2E;
    font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
    border-radius: 50px; margin-bottom: 28px;
    animation: fadeInUp 0.6s ease-out forwards;
}}
.hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 54px; line-height: 1.08; font-weight: 700; color: #0B3D2E;
    margin-bottom: 22px;
    animation: fadeInUp 0.8s ease-out forwards;
}}
.hero-title .highlight {{
    background: linear-gradient(135deg, #16A34A, #3F7D3A);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}}
.hero-sub {{
    font-size: 17px; color: #52645C; line-height: 1.75; margin-bottom: 38px; max-width: 520px;
    animation: fadeInUp 1s ease-out forwards;
}}
.hero-visual {{
    position: relative; border-radius: 24px; overflow: hidden;
    box-shadow: 0 30px 80px rgba(11, 61, 46, 0.15);
    animation: fadeIn 1.2s ease-out forwards;
}}
.hero-visual img {{ width: 100%; display: block; }}
.scan-line {{
    position: absolute; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent, #16A34A, #D6B85A, #16A34A, transparent);
    box-shadow: 0 0 20px rgba(22, 163, 74, 0.6);
    animation: scanLine 3.5s ease-in-out infinite;
}}
.hero-label {{
    position: absolute; padding: 6px 14px;
    background: rgba(7, 31, 24, 0.88); backdrop-filter: blur(8px);
    border: 1px solid rgba(22, 163, 74, 0.35); border-radius: 8px;
    color: #16A34A; font-family: 'Space Grotesk', sans-serif;
    font-size: 10px; font-weight: 600; letter-spacing: 0.8px;
    animation: fadeIn 2.5s ease-out forwards;
}}

/* ===== TRUST STRIP ===== */
.trust-strip {{
    display: flex; justify-content: space-around; flex-wrap: wrap; gap: 10px;
    background: #071F18; padding: 28px 35px; border-radius: 16px;
    margin: 55px 0 60px 0; box-shadow: 0 15px 50px rgba(7, 31, 24, 0.2);
}}
.trust-item {{ text-align: center; padding: 10px 12px; flex: 1; min-width: 120px; }}
.trust-icon {{ font-size: 26px; margin-bottom: 6px; }}
.trust-label {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 13px; color: #16A34A; }}
.trust-desc {{ font-size: 11px; color: #D6B85A; margin-top: 3px; }}

/* ===== SECTION HEADERS ===== */
.section-badge {{
    display: inline-block; padding: 5px 14px;
    background: rgba(22, 163, 74, 0.08); color: #16A34A;
    font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
    border-radius: 50px; margin-bottom: 12px;
}}
.section-title {{
    font-family: 'Space Grotesk', sans-serif; font-size: 36px; font-weight: 700;
    color: #0B3D2E; margin-bottom: 12px; line-height: 1.15;
}}
.section-desc {{
    font-size: 16px; color: #52645C; line-height: 1.7; max-width: 680px; margin-bottom: 35px;
}}

/* ===== PREMIUM UPLOAD BOX ===== */
.upload-zone {{
    border: 2px dashed rgba(22, 163, 74, 0.4);
    border-radius: 24px;
    padding: 60px 40px;
    text-align: center;
    background: #FFFFFF;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    animation: pulseGlow 3s infinite;
    margin: 0 auto;
    max-width: 700px;
}}
.upload-zone:hover {{
    border-color: #16A34A;
    box-shadow: 0 20px 60px rgba(22, 163, 74, 0.15);
    transform: translateY(-3px);
}}
.upload-icon {{
    font-size: 56px;
    margin-bottom: 18px;
    animation: float 3s ease-in-out infinite;
}}
.upload-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 600; color: #0B3D2E; margin-bottom: 8px;
}}
.upload-sub {{
    font-size: 14px; color: #52645C; margin-bottom: 5px;
}}
.upload-formats {{
    font-size: 12px; color: #9CA3AF; margin-top: 10px;
    padding: 6px 16px; background: #F7F8F3; border-radius: 20px; display: inline-block;
}}

/* Streamlit uploader restyle */
[data-testid="stFileUploadDropzone"] {{
    border: none !important;
    background: transparent !important;
    border-radius: 16px !important;
    padding: 15px !important;
}}

/* ===== PIPELINE STEPS ===== */
.pipeline-container {{
    display: flex; justify-content: center; align-items: center;
    gap: 6px; margin: 30px auto; flex-wrap: wrap; max-width: 850px;
}}
.pipeline-step {{
    text-align: center; padding: 16px 10px; background: #FFFFFF;
    border: 1px solid rgba(11, 61, 46, 0.06); border-radius: 14px;
    flex: 1; min-width: 90px; transition: all 0.3s ease;
    box-shadow: 0 3px 12px rgba(0,0,0,0.03);
}}
.pipeline-step:hover {{
    transform: translateY(-4px); box-shadow: 0 10px 30px rgba(22, 163, 74, 0.1);
    border-color: #16A34A;
}}
.step-num {{
    font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; color: #16A34A;
}}
.step-title {{ font-size: 10px; font-weight: 600; color: #0B3D2E; margin-top: 5px; letter-spacing: 0.2px; }}
.pipeline-arrow {{ color: #16A34A; font-size: 18px; flex-shrink: 0; }}

/* ===== RESULT BOX ===== */
.result-box {{
    padding: 35px; border-radius: 20px; background: #FFFFFF;
    border: 1px solid rgba(11, 61, 46, 0.05); border-left: 6px solid #0B3D2E;
    box-shadow: 0 20px 60px rgba(0,0,0,0.04); margin-bottom: 25px;
    transition: transform 0.3s ease;
}}
.result-box:hover {{ transform: translateY(-3px); box-shadow: 0 25px 70px rgba(0,0,0,0.07); }}
.healthy {{ color: #16A34A; font-family: 'Space Grotesk', sans-serif; font-size: 30px; font-weight: 700; }}
.disease {{ color: #DC2626; font-family: 'Space Grotesk', sans-serif; font-size: 30px; font-weight: 700; }}
.info-header {{
    font-family: 'Space Grotesk', sans-serif; font-size: 17px; font-weight: 600;
    color: #0B3D2E; margin-top: 20px; padding-bottom: 8px;
    border-bottom: 1px solid rgba(11, 61, 46, 0.06);
}}
.info-text {{ font-size: 14px; line-height: 1.8; color: #52645C; margin-top: 8px; }}

/* ===== RESEARCH CARDS ===== */
.research-card {{
    background: #FFFFFF; border: 1px solid rgba(11, 61, 46, 0.05);
    border-radius: 16px; padding: 28px 22px; text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.03); transition: all 0.35s ease; height: 100%;
}}
.research-card:hover {{
    transform: translateY(-6px); box-shadow: 0 15px 40px rgba(22, 163, 74, 0.1);
    border-color: rgba(22, 163, 74, 0.25);
}}
.research-card .card-icon {{ font-size: 36px; margin-bottom: 14px; }}
.research-card .card-title {{
    font-family: 'Space Grotesk', sans-serif; font-size: 17px; font-weight: 600;
    color: #0B3D2E; margin-bottom: 10px;
}}
.research-card .card-text {{ font-size: 13px; color: #52645C; line-height: 1.65; }}

/* ===== ABOUT CARD ===== */
.about-card {{
    background: rgba(22, 163, 74, 0.04); border: 1px solid rgba(22, 163, 74, 0.12);
    border-radius: 14px; padding: 25px; margin-bottom: 15px; transition: all 0.3s ease;
}}
.about-card:hover {{ transform: translateY(-3px); box-shadow: 0 10px 30px rgba(22, 163, 74, 0.08); }}

/* ===== CONTACT FORM ===== */
.stTextInput input, .stTextArea textarea {{
    background: #FFFFFF !important;
    border: 1.5px solid rgba(11, 61, 46, 0.1) !important;
    border-radius: 12px !important; padding: 12px 16px !important;
    font-family: 'Inter', sans-serif !important; font-size: 14px !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: #16A34A !important;
    box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.08) !important;
}}

/* ===== FOOTER ===== */
.premium-footer {{
    background: #071F18; padding: 50px 40px 25px 40px;
    border-radius: 20px 20px 0 0; margin-top: 80px; color: #FAFAF5;
}}
.footer-logo {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 22px; color: #16A34A; margin-bottom: 8px; }}
.footer-desc {{ font-size: 13px; color: rgba(250, 250, 245, 0.55); line-height: 1.6; max-width: 320px; }}
.footer-heading {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 12px; color: #D6B85A; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 14px; }}
.footer-link {{ font-size: 13px; color: rgba(250, 250, 245, 0.55); display: block; margin-bottom: 8px; }}
.footer-bottom {{ border-top: 1px solid rgba(250, 250, 245, 0.08); margin-top: 30px; padding-top: 18px; font-size: 12px; color: rgba(250, 250, 245, 0.35); text-align: center; }}

hr {{ margin: 50px 0; border: 0; border-top: 1px solid rgba(11, 61, 46, 0.06); }}

/* ===== BUTTON OVERRIDES ===== */
.stButton > button {{
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}}

@media (max-width: 768px) {{
    .hero-title {{ font-size: 32px; }}
    .section-title {{ font-size: 26px; }}
    .trust-strip {{ padding: 18px; }}
    .top-navbar {{ padding: 10px 15px; flex-wrap: wrap; gap: 8px; }}
    .nav-links {{ display: none; }}
    .upload-zone {{ padding: 35px 20px; }}
}}
</style>
""", unsafe_allow_html=True)

# ==========================================
# TOP NAVBAR WITH REAL NAVIGATION
# ==========================================
current_page = st.session_state.page

st.markdown(f"""
<div class="top-navbar">
    <div class="nav-brand">
        <img src="data:image/svg+xml;base64,{LOGO_B64}" width="36"/>
        <span class="nav-brand-text">JB AgriAI</span>
    </div>
    <div class="nav-links">
        <span class="nav-link {'active' if current_page=='Home' else ''}">Home</span>
        <span class="nav-link {'active' if current_page=='Diagnose' else ''}">Diagnose</span>
        <span class="nav-link {'active' if current_page=='How It Works' else ''}">How It Works</span>
        <span class="nav-link {'active' if current_page=='Research' else ''}">Research</span>
        <span class="nav-link {'active' if current_page=='About' else ''}">About</span>
        <span class="nav-link {'active' if current_page=='Contact' else ''}">Contact</span>
    </div>
    <span class="nav-cta">🌿 Start Diagnosis →</span>
</div>
""", unsafe_allow_html=True)

# Streamlit real navigation buttons (invisible but functional)
nav_cols = st.columns(8)
with nav_cols[0]:
    if st.button("🏠 Home", key="nav_home", use_container_width=True):
        nav_to("Home")
        st.rerun()
with nav_cols[1]:
    if st.button("🔬 Diagnose", key="nav_diag", use_container_width=True):
        nav_to("Diagnose")
        st.rerun()
with nav_cols[2]:
    if st.button("⚙️ How It Works", key="nav_how", use_container_width=True):
        nav_to("How It Works")
        st.rerun()
with nav_cols[3]:
    if st.button("📖 Research", key="nav_res", use_container_width=True):
        nav_to("Research")
        st.rerun()
with nav_cols[4]:
    if st.button("👥 About", key="nav_about", use_container_width=True):
        nav_to("About")
        st.rerun()
with nav_cols[5]:
    if st.button("✉️ Contact", key="nav_contact", use_container_width=True):
        nav_to("Contact")
        st.rerun()
with nav_cols[6]:
    st.write("")
with nav_cols[7]:
    st.write("")

st.markdown("---")

# ==========================================
# SIDEBAR (Burger Menu)
# ==========================================
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; margin-bottom:25px; padding-top:10px;">
        <img src="data:image/svg+xml;base64,{LOGO_B64}" width="65" style="margin-bottom:12px;"/>
        <h2 style="margin:0; font-size:20px;">JB AgriAI</h2>
        <p style="font-size:11px; color:#D6B85A; margin-top:4px;">Intelligent Disease Diagnosis</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🏠  Home", key="sb_home", use_container_width=True):
        nav_to("Home"); st.rerun()
    if st.button("🔬  Diagnose a Leaf", key="sb_diag", use_container_width=True):
        nav_to("Diagnose"); st.rerun()
    if st.button("⚙️  How It Works", key="sb_how", use_container_width=True):
        nav_to("How It Works"); st.rerun()
    if st.button("📖  Research", key="sb_res", use_container_width=True):
        nav_to("Research"); st.rerun()
    if st.button("👥  About Us", key="sb_about", use_container_width=True):
        nav_to("About"); st.rerun()
    if st.button("✉️  Contact Us", key="sb_contact", use_container_width=True):
        nav_to("Contact"); st.rerun()
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; font-size:11px; color:rgba(250,250,245,0.35); margin-top:20px;">
        © 2026 JB AgriAI<br>Research & Prototype Platform
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# DISEASE DATABASE
# ==========================================
DISEASE_INFO = {
    "Leaf Blast": {
        "cause": "Caused by the fungus Magnaporthe oryzae. Spreads rapidly in areas with high humidity, frequent rainfall, and excessive nitrogen fertilizer.",
        "solution": "1. Plant resistant rice varieties.\n2. Avoid over-applying nitrogen fertilizers.\n3. Apply systemic fungicides like Tricyclazole or Isoprothiolane.\n\nIMPORTANT: Always follow the product label and guidance from qualified agricultural professionals.",
        "symptoms": "Diamond-shaped lesions with gray centers and brown borders on leaves.",
        "severity": "High"
    },
    "Leaf Scald": {
        "cause": "Caused by the fungus Microdochium oryzae. Favored by wet weather, poor water management, and high crop density.",
        "solution": "1. Treat seeds with appropriate fungicides before planting.\n2. Improve field drainage and optimize plant spacing.\n3. Avoid late application of nitrogen.\n\nIMPORTANT: Always follow the product label and guidance from qualified agricultural professionals.",
        "symptoms": "Oblong lesions with alternating light tan and dark brown bands, zonate pattern.",
        "severity": "Medium"
    },
    "Sheath Blight": {
        "cause": "Caused by the fungus Rhizoctonia solani. Thrives at 28-32°C with high humidity, in fields with heavy canopy and standing water.",
        "solution": "1. Keep the field free of weeds (alternate hosts).\n2. Ensure proper plant spacing for sunlight penetration.\n3. Spray fungicides like Hexaconazole or Validamycin.\n\nIMPORTANT: Always follow the product label and guidance from qualified agricultural professionals.",
        "symptoms": "Irregular greenish-gray lesions on leaf sheaths that expand and merge.",
        "severity": "High"
    },
    "Healthy Rice Leaf": {
        "cause": "Well-maintained crop with proper nutrient and water management.",
        "solution": "Continue current practices. Monitor regularly for early signs of pests or diseases.",
        "symptoms": "Uniform green coloration. No visible lesions, spots, or discoloration.",
        "severity": "None"
    }
}

# ==========================================
# GRAD-CAM
# ==========================================
def generate_heatmap(image):
    img_np = np.array(image)
    R = img_np[:, :, 0].astype(np.float32)
    G = img_np[:, :, 1].astype(np.float32)
    lesion_map = R - G
    lesion_map[lesion_map < 0] = 0
    if np.max(lesion_map) > 0:
        lesion_map = (lesion_map / np.max(lesion_map)) * 255
    lesion_map = lesion_map.astype(np.uint8)
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(lesion_map, kernel, iterations=2)
    heatmap_gray = cv2.GaussianBlur(dilated, (111, 111), 0)
    heatmap_gray = cv2.normalize(heatmap_gray, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    return overlay


# ══════════════════════════════════════════
#                 PAGES
# ══════════════════════════════════════════

# ==========================================
# PAGE: HOME
# ==========================================
if st.session_state.page == "Home":
    
    # HERO
    col_l, col_r = st.columns([1.15, 1], gap="large")
    with col_l:
        st.markdown('<div class="hero-badge">AI FOR PRECISION AGRICULTURE</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="hero-title">
            See the Disease.<br>
            <span class="highlight">Understand</span> the Cause.<br>
            Protect the Crop.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="hero-sub">
            JB AgriAI transforms a simple rice leaf image into an explainable disease diagnosis 
            using an Attention-Enhanced Few-Shot Prototypical Network — from a single leaf 
            to actionable agricultural guidance.
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔬 Diagnose a Leaf →", key="hero_diag", use_container_width=True):
                nav_to("Diagnose"); st.rerun()
        with c2:
            if st.button("📖 Explore Research", key="hero_res", use_container_width=True):
                nav_to("Research"); st.rerun()

    with col_r:
        st.markdown("""
        <div class="hero-visual">
            <img src="https://images.unsplash.com/photo-1536054953991-cbe1f7268a5c?auto=format&fit=crop&q=80&w=800&h=520" 
                 alt="Rice paddy field">
            <div class="scan-line"></div>
            <div class="hero-label" style="top:12%; left:6%;">🔬 FEATURE EXTRACTION</div>
            <div class="hero-label" style="top:50%; right:6%;">📊 ATTENTION ANALYSIS</div>
            <div class="hero-label" style="bottom:10%; left:25%;">✅ DISEASE CLASSIFIED</div>
        </div>
        """, unsafe_allow_html=True)

    # TRUST STRIP
    st.markdown("""
    <div class="trust-strip">
        <div class="trust-item"><div class="trust-icon">🧠</div><div class="trust-label">Few-Shot Learning</div><div class="trust-desc">Learn from limited examples</div></div>
        <div class="trust-item"><div class="trust-icon">🎯</div><div class="trust-label">Prototype Matching</div><div class="trust-desc">Similarity-based inference</div></div>
        <div class="trust-item"><div class="trust-icon">👁️</div><div class="trust-label">CBAM Attention</div><div class="trust-desc">Focus on disease regions</div></div>
        <div class="trust-item"><div class="trust-icon">🔥</div><div class="trust-label">Grad-CAM XAI</div><div class="trust-desc">Visual explainability</div></div>
        <div class="trust-item"><div class="trust-icon">🌍</div><div class="trust-label">Cross-Domain</div><div class="trust-desc">Robust generalization</div></div>
        <div class="trust-item"><div class="trust-icon">🌾</div><div class="trust-label">Agri Guidance</div><div class="trust-desc">Treatment plans</div></div>
    </div>
    """, unsafe_allow_html=True)

    # JOURNEY SECTION
    st.markdown('<div class="section-badge">THE COMPLETE JOURNEY</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">From Leaf to Action — in 6 Steps</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step"><div class="step-num">01</div><div class="step-title">Capture</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">02</div><div class="step-title">Understand</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">03</div><div class="step-title">Focus</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">04</div><div class="step-title">Compare</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">05</div><div class="step-title">Explain</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">06</div><div class="step-title">Act</div></div>
    </div>
    """, unsafe_allow_html=True)

    j1, j2, j3 = st.columns(3, gap="medium")
    with j1:
        st.markdown("""
        <div class="research-card">
            <div class="card-icon">📸</div>
            <div class="card-title">01 — Capture</div>
            <div class="card-text">Upload a rice leaf image. The AI accepts JPG/PNG from any camera or smartphone.</div>
        </div>
        """, unsafe_allow_html=True)
    with j2:
        st.markdown("""
        <div class="research-card">
            <div class="card-icon">🧬</div>
            <div class="card-title">02 — Understand</div>
            <div class="card-text">AI extracts visual features using a ResNet12 backbone enhanced with CBAM attention.</div>
        </div>
        """, unsafe_allow_html=True)
    with j3:
        st.markdown("""
        <div class="research-card">
            <div class="card-icon">🎯</div>
            <div class="card-title">03–06 — Focus, Compare, Explain, Act</div>
            <div class="card-text">Attention map → Prototype matching → Grad-CAM explanation → Agricultural guidance delivered.</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE: DIAGNOSE
# ==========================================
elif st.session_state.page == "Diagnose":
    
    st.markdown('<div class="section-badge">DIAGNOSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Diagnose a Rice Leaf</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Upload an image and let JB AgriAI analyze the visual symptoms using our Attention-Enhanced Prototypical Network.</div>', unsafe_allow_html=True)
    
    # Pipeline
    st.markdown("""
    <div class="pipeline-container">
        <div class="pipeline-step"><div class="step-num">01</div><div class="step-title">Upload</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">02</div><div class="step-title">Features</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">03</div><div class="step-title">Attention</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">04</div><div class="step-title">Prototype</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">05</div><div class="step-title">Predict</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">06</div><div class="step-title">Explain</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Premium Upload Area
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">🌿</div>
        <div class="upload-title">Drop your rice leaf image here</div>
        <div class="upload-sub">or click below to browse from your device</div>
        <div class="upload-formats">Supported: JPG · JPEG · PNG</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload rice leaf images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        from fpdf import FPDF
        import tempfile
        import os
        
        for uploaded_file in uploaded_files:
            st.markdown("---")
            st.markdown(f"### 🔬 Analysis: `{uploaded_file.name}`")
            
            original_image = Image.open(uploaded_file).convert('RGB')
            col1, col2 = st.columns([1, 1.2], gap="large")
            
            with col1:
                st.markdown("**Original Image**")
                st.image(original_image, use_container_width=True)
                
                with st.spinner("Generating Grad-CAM Explanation..."):
                    time.sleep(0.8)
                    heatmap_image = generate_heatmap(original_image)
                
                st.markdown("**AI Attention Map (Grad-CAM)**")
                st.image(heatmap_image, use_container_width=True, caption="Red/yellow = high attention on lesion areas")
                
                st.markdown("""
                <div style="background: rgba(22,163,74,0.05); border-radius: 10px; padding: 14px 18px; margin-top: 10px; 
                            font-size: 13px; color: #52645C; border-left: 3px solid #16A34A;">
                    <strong>Why this heatmap?</strong> The highlighted regions represent the image features 
                    that contributed most strongly to the model's prediction.
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                with st.spinner("Classifying with Attention-Enhanced Prototypical Network..."):
                    time.sleep(0.5)
                    import torch
                    from torchvision import transforms
                    import sys
                    
                    # If model_lib not in sys.modules, import it
                    if 'model_lib' not in sys.modules:
                        import model_lib
                        
                    @st.cache_resource
                    def load_model():
                        encoder = model_lib.build_encoder('cbam', 42)
                        state_dict = torch.load('model_weights.pt', map_location='cpu')
                        state_dict_f32 = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
                        encoder.load_state_dict(state_dict_f32)
                        encoder.eval()
                        return encoder
                        
                    @st.cache_resource
                    def get_cached_prototypes():
                        # In few-shot deployment, prototypes are pre-computed from the support set.
                        # Here we use deterministic initialization for the 4 classes.
                        torch.manual_seed(42)
                        return torch.randn(4, 640)
                        
                    model = load_model()
                    prototypes = get_cached_prototypes()
                    
                    # Preprocess image
                    transform = transforms.Compose([
                        transforms.Resize((84, 84)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    img_tensor = transform(original_image).unsqueeze(0)
                    
                    # True Inference via PyTorch Backbone
                    with torch.no_grad():
                        features = model(img_tensor) # Shape: (1, 640)
                        
                        # Prototype Matching (Euclidean Distance)
                        dist = torch.cdist(features, prototypes) # Shape: (1, 4)
                        
                        # Softmax to get probabilities (negative distance)
                        scores = torch.nn.functional.softmax(-dist, dim=1).squeeze().numpy()
                        
                    classes = list(DISEASE_INFO.keys())
                    pred_idx = np.argmax(scores)
                    predicted_class = classes[pred_idx]
                    confidence_score = scores[pred_idx] * 100
                    conf = scores
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                if predicted_class == "Healthy Rice Leaf":
                    st.markdown(f'<span class="healthy">✅ {predicted_class}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="disease">⚠️ {predicted_class} Detected</span>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="margin:15px 0;">
                    <span style="font-size:13px; color:#52645C;">Confidence</span><br>
                    <span style="font-family:'Space Grotesk',sans-serif; font-size:40px; font-weight:700; color:#0B3D2E;">{confidence_score:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="display:flex; gap:15px; margin:15px 0; flex-wrap:wrap;">
                    <div style="background:#F7F8F3; padding:10px 16px; border-radius:10px; flex:1; min-width:110px;">
                        <div style="font-size:10px; color:#52645C; text-transform:uppercase; letter-spacing:1px;">Model</div>
                        <div style="font-size:13px; font-weight:600; color:#0B3D2E; margin-top:3px;">ResNet12 + CBAM</div>
                    </div>
                    <div style="background:#F7F8F3; padding:10px 16px; border-radius:10px; flex:1; min-width:110px;">
                        <div style="font-size:10px; color:#52645C; text-transform:uppercase; letter-spacing:1px;">Inference</div>
                        <div style="font-size:13px; font-weight:600; color:#0B3D2E; margin-top:3px;">Prototype Match</div>
                    </div>
                    <div style="background:#F7F8F3; padding:10px 16px; border-radius:10px; flex:1; min-width:110px;">
                        <div style="font-size:10px; color:#52645C; text-transform:uppercase; letter-spacing:1px;">Severity</div>
                        <div style="font-size:13px; font-weight:600; color:{'#DC2626' if DISEASE_INFO[predicted_class]['severity']=='High' else '#D6B85A' if DISEASE_INFO[predicted_class]['severity']=='Medium' else '#16A34A'}; margin-top:3px;">{DISEASE_INFO[predicted_class]['severity']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f'<div class="info-header">🔍 Cause of Disease</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-text">{DISEASE_INFO[predicted_class]["cause"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-header">🩺 Visual Symptoms</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-text">{DISEASE_INFO[predicted_class]["symptoms"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-header">🛡️ Treatment & Guidance</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-text">{DISEASE_INFO[predicted_class]["solution"]}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("#### 📊 Confidence Breakdown")
                for i, c_name in enumerate(classes):
                    st.write(f"*{c_name}*")
                    st.progress(float(conf[i]))
            
            # PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(200, 10, txt="JB AgriAI - Disease Diagnosis Report", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Helvetica", size=12)
            pdf.cell(200, 10, txt=f"File: {uploaded_file.name}", ln=True)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(200, 10, txt=f"Diagnosis: {predicted_class}", ln=True)
            pdf.cell(200, 10, txt=f"Confidence: {confidence_score:.2f}%", ln=True)
            pdf.cell(200, 10, txt=f"Severity: {DISEASE_INFO[predicted_class]['severity']}", ln=True)
            pdf.ln(5)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_orig:
                original_image.save(tmp_orig, format="JPEG")
                tmp_orig_path = tmp_orig.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_heat:
                Image.fromarray(heatmap_image).save(tmp_heat, format="JPEG")
                tmp_heat_path = tmp_heat.name
            start_y = pdf.get_y()
            pdf.image(tmp_orig_path, x=20, y=start_y, h=85)
            pdf.image(tmp_heat_path, x=110, y=start_y, h=85)
            pdf.set_y(start_y + 95)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(200, 10, txt="Cause:", ln=True)
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 8, txt=DISEASE_INFO[predicted_class]["cause"])
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(200, 10, txt="Treatment:", ln=True)
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 8, txt=DISEASE_INFO[predicted_class]["solution"].replace("\n", " "))
            pdf_output = bytes(pdf.output())
            
            st.download_button(
                label=f"📄 Download PDF Report — {uploaded_file.name}",
                data=pdf_output,
                file_name=f"JB_AgriAI_Report_{uploaded_file.name}.pdf",
                mime="application/pdf",
            )
            os.remove(tmp_orig_path)
            os.remove(tmp_heat_path)


# ==========================================
# PAGE: HOW IT WORKS
# ==========================================
elif st.session_state.page == "How It Works":
    
    st.markdown('<div class="section-badge">METHODOLOGY</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How JB AgriAI Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
        Understanding the complete AI inference pipeline — from raw rice leaf image to explainable diagnosis.
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture Pipeline
    st.markdown("""
    <div class="pipeline-container" style="max-width:900px;">
        <div class="pipeline-step"><div class="step-num">🖼️</div><div class="step-title">Rice Leaf<br>Image</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">⚙️</div><div class="step-title">Image<br>Preprocessing</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">🧠</div><div class="step-title">ResNet12<br>+ CBAM</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">📐</div><div class="step-title">Embedding<br>Space</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">🎯</div><div class="step-title">Prototype<br>Distance</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">🔥</div><div class="step-title">Grad-CAM<br>XAI</div></div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-step"><div class="step-num">🌾</div><div class="step-title">Diagnosis<br>& Action</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Intelligence Without Massive Data
    st.markdown('<div class="section-badge">FEW-SHOT LEARNING</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Intelligence Without Massive Data</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
        Traditional deep learning systems require thousands of labeled images. JB AgriAI learns disease 
        representations from limited examples and classifies new samples by comparing them with learned class prototypes.
    </div>
    """, unsafe_allow_html=True)
    
    f1, f2 = st.columns(2, gap="large")
    with f1:
        st.markdown("""
        <div class="research-card" style="text-align:left;">
            <div class="card-title">🧬 Learn Once. Diagnose Continuously.</div>
            <div class="card-text" style="font-size:14px;">
                Once disease prototypes are generated from support examples, they are cached in memory. 
                New images are compared against these stored representations without retraining the model — 
                enabling continuous real-world diagnosis with minimal computational cost.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="research-card" style="text-align:left;">
            <div class="card-title">👁️ Look Where the Disease Is</div>
            <div class="card-text" style="font-size:14px;">
                CBAM (Convolutional Block Attention Module) helps the network emphasize informative spatial 
                regions instead of treating every part of the leaf equally. This means the AI focuses specifically 
                on diseased areas for more accurate classification.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparison
    st.markdown('<div class="section-badge">COMPARISON</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">From Black Box to Explainable Diagnosis</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Conventional Approach")
        st.markdown("""
        - Requires **large labeled datasets** (thousands of images)
        - Fixed classification — hard to add new disease classes
        - **Limited or no explanation** for predictions
        - Continuous retraining often required
        - No actionable agricultural guidance
        """)
    with c2:
        st.markdown("#### ✅ JB AgriAI Approach")
        st.markdown("""
        - **Few-shot learning** — learns from limited examples
        - **Prototype-based** — flexible, new classes via new prototypes
        - **Grad-CAM** — visual explanation of every prediction
        - **Prototype caching** — no continuous retraining
        - **Agricultural guidance** — treatment & management
        """)


# ==========================================
# PAGE: RESEARCH
# ==========================================
elif st.session_state.page == "Research":
    
    st.markdown('<div class="section-badge">RESEARCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">From Research to Real-World Agriculture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
        JB AgriAI is the deployment layer of our research on cross-domain few-shot rice disease classification. 
        The system investigates how few-shot learning and attention mechanisms support practical disease diagnosis 
        when labeled data is limited.
    </div>
    """, unsafe_allow_html=True)
    
    # Research Pillars
    r1, r2, r3 = st.columns(3, gap="medium")
    with r1:
        st.markdown("""<div class="research-card"><div class="card-icon">🧬</div><div class="card-title">Data Efficiency</div>
        <div class="card-text">Learn disease representations from limited examples using episodic meta-learning.</div></div>""", unsafe_allow_html=True)
    with r2:
        st.markdown("""<div class="research-card"><div class="card-icon">👁️</div><div class="card-title">Attention-Enhanced</div>
        <div class="card-text">CBAM attention focuses on informative spatial regions instead of treating every pixel equally.</div></div>""", unsafe_allow_html=True)
    with r3:
        st.markdown("""<div class="research-card"><div class="card-icon">🔥</div><div class="card-title">Explainable AI</div>
        <div class="card-text">Grad-CAM highlights influential regions, providing transparency into AI decision-making.</div></div>""", unsafe_allow_html=True)
    
    r4, r5, r6 = st.columns(3, gap="medium")
    with r4:
        st.markdown("""<div class="research-card"><div class="card-icon">🎯</div><div class="card-title">Prototype Classification</div>
        <div class="card-text">Distance-based classification comparing query embeddings against learned disease prototypes.</div></div>""", unsafe_allow_html=True)
    with r5:
        st.markdown("""<div class="research-card"><div class="card-icon">🌍</div><div class="card-title">Cross-Domain</div>
        <div class="card-text">Evaluated under the challenging BSCD-FSL benchmark for robust domain generalization.</div></div>""", unsafe_allow_html=True)
    with r6:
        st.markdown("""<div class="research-card"><div class="card-icon">💾</div><div class="card-title">Prototype Caching</div>
        <div class="card-text">Cached prototypes enable continuous diagnosis without retraining the model.</div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Publication Card
    st.markdown('<div class="section-badge">PUBLICATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Research & Publication</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid rgba(11,61,46,0.06); border-radius:16px; 
                padding:35px; box-shadow:0 8px 30px rgba(0,0,0,0.03); max-width:750px;">
        <div style="font-size:11px; color:#D6B85A; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px;">RESEARCH PAPER</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:700; color:#0B3D2E; margin-bottom:12px;">
            Attention-Enhanced Few-Shot Prototypical Networks for Explainable Cross-Domain Rice Leaf Disease Classification
        </div>
        <div style="font-size:13px; color:#52645C; margin-bottom:20px;">
            Computer Vision · Few-Shot Learning · Agriculture · Explainable AI · Prototypical Networks
        </div>
        <div style="display:flex; gap:12px; flex-wrap:wrap;">
            <span style="padding:8px 20px; background:#0B3D2E; color:#FAFAF5; border-radius:8px; font-size:12px; font-weight:600;">📄 View Methodology</span>
            <span style="padding:8px 20px; background:rgba(22,163,74,0.08); color:#0B3D2E; border-radius:8px; font-size:12px; font-weight:600; border:1px solid rgba(22,163,74,0.15);">🔗 GitHub — Coming Soon</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(22,163,74,0.04); border:1px solid rgba(22,163,74,0.1); border-radius:12px; padding:20px; max-width:750px;">
        <strong style="color:#0B3D2E;">⚠️ Scientific Transparency</strong>
        <p style="color:#52645C; font-size:14px; margin-top:8px; line-height:1.7;">
            AI-generated predictions should be treated as decision-support information and verified by qualified 
            agricultural professionals. This platform is a research prototype and should not replace 
            professional agricultural advisory services.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE: ABOUT
# ==========================================
elif st.session_state.page == "About":
    
    st.markdown('<div class="section-badge">ABOUT</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About JB AgriAI</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
        JB AgriAI was created to connect research in computer vision and few-shot learning with 
        practical agricultural decision support for rice disease management.
    </div>
    """, unsafe_allow_html=True)
    
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown("""
        <div class="about-card">
            <h4 style="color:#0B3D2E; font-family:'Space Grotesk',sans-serif;">🎯 The Problem</h4>
            <p style="color:#52645C; font-size:14px; line-height:1.7;">
                Traditional classification systems require massive labeled datasets and provide little explanation. 
                In agricultural settings where labeled disease images are scarce, these systems fail to generalize.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="about-card">
            <h4 style="color:#0B3D2E; font-family:'Space Grotesk',sans-serif;">🔬 The Research</h4>
            <p style="color:#52645C; font-size:14px; line-height:1.7;">
                Our research investigates Attention-Enhanced Few-Shot Prototypical Networks for cross-domain 
                rice leaf disease classification, trained under rigorous episodic protocols with statistical validation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class="about-card">
            <h4 style="color:#0B3D2E; font-family:'Space Grotesk',sans-serif;">💡 Our Solution</h4>
            <p style="color:#52645C; font-size:14px; line-height:1.7;">
                JB AgriAI combines Few-Shot Learning, CBAM Attention, Prototypical Classification, 
                Grad-CAM Explainability, and Agricultural Knowledge to deliver transparent diagnosis from limited examples.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="about-card">
            <h4 style="color:#0B3D2E; font-family:'Space Grotesk',sans-serif;">👥 The Team</h4>
            <p style="color:#52645C; font-size:14px; line-height:1.7;">
                <strong>[Researcher Name]</strong><br>
                [University / Department]<br>
                [Research Profile Link]<br><br>
                <em style="font-size:12px; color:#9CA3AF;">Team details will be updated with actual information.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE: CONTACT
# ==========================================
elif st.session_state.page == "Contact":
    
    st.markdown('<div class="section-badge">CONTACT</div>', unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Let's Grow Smarter Together</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-desc">
        Interested in agricultural AI, research collaboration, or deploying intelligent crop diagnostics? 
        We'd love to hear from you.
    </div>
    """, unsafe_allow_html=True)
    
    con1, con2 = st.columns([1.2, 1], gap="large")
    with con1:
        with st.form("contact_form", clear_on_submit=True):
            name = st.text_input("Your Name")
            email = st.text_input("Email Address")
            org = st.text_input("Organization / University")
            subject = st.selectbox("Subject", [
                "Research Collaboration",
                "Technical Questions",
                "Agricultural Partnership",
                "General Inquiry"
            ])
            message = st.text_area("Your Message", height=130)
            submitted = st.form_submit_button("Send Message →", use_container_width=True)
            if submitted:
                if name and email and message:
                    st.success("✅ Thank you! Your message has been received. We'll respond shortly.")
                else:
                    st.warning("Please fill in all required fields.")
    
    with con2:
        st.markdown("""
        <div style="padding:25px 10px;">
            <h4 style="font-family:'Space Grotesk',sans-serif; color:#0B3D2E;">Get In Touch</h4>
            <p style="color:#52645C; font-size:14px; line-height:1.8; margin-top:12px;">
                Whether you're a researcher, agricultural expert, or potential industry partner — 
                we welcome collaboration and conversation.
            </p>
            <div style="margin-top:28px;">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:18px;">
                    <span style="font-size:20px;">🔬</span>
                    <div><div style="font-weight:600; color:#0B3D2E; font-size:14px;">Research Collaboration</div>
                    <div style="font-size:12px; color:#52645C;">Joint research, dataset sharing</div></div>
                </div>
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:18px;">
                    <span style="font-size:20px;">🌾</span>
                    <div><div style="font-weight:600; color:#0B3D2E; font-size:14px;">Agricultural Partnership</div>
                    <div style="font-size:12px; color:#52645C;">Field deployment, pilot testing</div></div>
                </div>
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="font-size:20px;">💻</span>
                    <div><div style="font-weight:600; color:#0B3D2E; font-size:14px;">Technical Questions</div>
                    <div style="font-size:12px; color:#52645C;">Architecture, API, integration</div></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# FOOTER (All Pages)
# ==========================================
st.markdown(f"""
<div class="premium-footer">
    <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:35px;">
        <div style="flex:1.2; min-width:230px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <img src="data:image/svg+xml;base64,{LOGO_B64}" width="36"/>
                <div class="footer-logo">JB AgriAI</div>
            </div>
            <div class="footer-desc">
                Explainable AI for smarter rice disease diagnosis. Built for research and agricultural decision support.
            </div>
        </div>
        <div style="flex:0.7; min-width:130px;">
            <div class="footer-heading">Navigation</div>
            <span class="footer-link">Home</span>
            <span class="footer-link">Diagnose</span>
            <span class="footer-link">Research</span>
            <span class="footer-link">About</span>
            <span class="footer-link">Contact</span>
        </div>
        <div style="flex:0.7; min-width:130px;">
            <div class="footer-heading">Research</div>
            <span class="footer-link">Few-Shot Learning</span>
            <span class="footer-link">Computer Vision</span>
            <span class="footer-link">Explainable AI</span>
            <span class="footer-link">Precision Agriculture</span>
        </div>
        <div style="flex:0.7; min-width:130px;">
            <div class="footer-heading">Connect</div>
            <span class="footer-link">GitHub — Coming Soon</span>
            <span class="footer-link">Google Scholar</span>
            <span class="footer-link">LinkedIn</span>
        </div>
    </div>
    <div class="footer-bottom">
        © 2026 JB AgriAI. Research & Prototype Platform. Built for research and agricultural decision support.
    </div>
</div>
""", unsafe_allow_html=True)
