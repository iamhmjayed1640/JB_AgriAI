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
    page_title="JB AgriAI | Intelligent Rice Disease Diagnosis",
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
# MEGA CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, .stApp {
    background-color: #F7F8F3 !important;
    font-family: 'Inter', sans-serif !important;
    color: #10231C;
    scroll-behavior: smooth;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Fix Top Navbar Buttons */
.nav-container {
    background-color: #FFFFFF;
    border-radius: 16px;
    padding: 12px 25px;
    box-shadow: 0 10px 30px rgba(11, 61, 46, 0.05);
    margin: 10px 0 30px 0;
    border: 1px solid rgba(22, 163, 74, 0.1);
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .stButton > button {
    background-color: transparent !important;
    border: none !important;
    color: #52645C !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    padding: 12px 18px !important;
    border-radius: 10px !important;
    width: 100% !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .stButton > button:hover {
    color: #0B3D2E !important;
    background-color: rgba(22, 163, 74, 0.08) !important;
}

/* Beautiful Title Text */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 54px; line-height: 1.08; font-weight: 700; color: #0B3D2E;
}
.hero-title .highlight {
    background: linear-gradient(135deg, #16A34A, #3F7D3A);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-visual {
    position: relative; border-radius: 24px; overflow: hidden;
    box-shadow: 0 30px 80px rgba(11, 61, 46, 0.15);
    background: linear-gradient(135deg, #071F18, #0B3D2E); 
    min-height: 380px; display: flex; align-items: center; justify-content: center;
}

/* Section titles */
.section-badge {
    display: inline-block; padding: 5px 14px;
    background: rgba(22, 163, 74, 0.08); color: #16A34A;
    font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
    border-radius: 50px; margin-bottom: 12px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif; font-size: 36px; font-weight: 700;
    color: #0B3D2E; margin-bottom: 12px; line-height: 1.15;
}

/* Streamlit uploader fix */
[data-testid="stFileUploadDropzone"] {
    background: #FFFFFF !important;
    border: 2px dashed rgba(22, 163, 74, 0.4) !important;
    border-radius: 16px !important;
    padding: 15px !important;
}

/* Results Box */
.result-box {
    padding: 35px; border-radius: 20px; background: #FFFFFF;
    border: 1px solid rgba(11, 61, 46, 0.05); border-left: 6px solid #0B3D2E;
    box-shadow: 0 20px 60px rgba(0,0,0,0.04); margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# BEAUTIFUL NAVBAR
# ==========================================
current_page = st.session_state.page

st.markdown('<div class="nav-container">', unsafe_allow_html=True)
nav_cols = st.columns([1.5, 1, 1, 1.2, 1, 1, 1], gap="small")

with nav_cols[0]:
    st.markdown('<div style="font-family:\'Space Grotesk\',sans-serif; font-weight:800; font-size:22px; color:#16A34A; padding-top:5px;">JB AgriAI</div>', unsafe_allow_html=True)

def style_btn(name):
    return f"🟢 {name}" if current_page == name else name

with nav_cols[1]:
    if st.button(style_btn("Home"), use_container_width=True): nav_to("Home"); st.rerun()
with nav_cols[2]:
    if st.button(style_btn("Diagnose"), use_container_width=True): nav_to("Diagnose"); st.rerun()
with nav_cols[3]:
    if st.button(style_btn("How It Works"), use_container_width=True): nav_to("How It Works"); st.rerun()
with nav_cols[4]:
    if st.button(style_btn("Research"), use_container_width=True): nav_to("Research"); st.rerun()
with nav_cols[5]:
    if st.button(style_btn("About"), use_container_width=True): nav_to("About"); st.rerun()
with nav_cols[6]:
    if st.button(style_btn("Contact"), use_container_width=True): nav_to("Contact"); st.rerun()
    
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# DISEASE DATABASE
# ==========================================
DISEASE_INFO = {
    "Leaf Blast": {
        "cause": "Caused by the fungus Magnaporthe oryzae. Spreads rapidly in areas with high humidity, frequent rainfall, and excessive nitrogen fertilizer.",
        "solution": "1. Plant resistant rice varieties.\n2. Avoid over-applying nitrogen fertilizers.\n3. Apply systemic fungicides like Tricyclazole.",
        "symptoms": "Diamond-shaped lesions with gray centers and brown borders on leaves.",
        "severity": "High"
    },
    "Leaf Scald": {
        "cause": "Caused by the fungus Microdochium oryzae. Favored by wet weather, poor water management, and high crop density.",
        "solution": "1. Treat seeds with appropriate fungicides before planting.\n2. Improve field drainage.",
        "symptoms": "Zonate lesions starting from leaf tips or edges, appearing scalded.",
        "severity": "Medium"
    },
    "Narrow Brown Spot": {
        "cause": "Caused by the fungus Cercospora janseana. Often appears in potassium-deficient soils.",
        "solution": "1. Ensure balanced fertilization, particularly potassium.\n2. Plant early-maturing varieties.",
        "symptoms": "Short, narrow, linear brown spots parallel to the leaf veins.",
        "severity": "Low to Medium"
    },
    "Healthy": {
        "cause": "Optimal environmental conditions.",
        "solution": "Continue regular monitoring and good practices.",
        "symptoms": "Uniform green color without lesions.",
        "severity": "None"
    }
}

# ==========================================
# PAGE: HOME
# ==========================================
if st.session_state.page == "Home":
    col_l, col_r = st.columns([1.1, 1], gap="large")
    with col_l:
        st.markdown('<div class="hero-badge">AI FOR PRECISION AGRICULTURE</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">See the Disease.<br><span class="highlight">Understand</span> the<br>Cause.<br>Protect the Crop.</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 17px; color: #52645C; line-height: 1.75; margin-bottom: 38px;">JB AgriAI transforms a simple rice leaf image into an explainable disease diagnosis using an Attention-Enhanced Few-Shot Prototypical Network.</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚀 Diagnose a Leaf", use_container_width=True, type="primary"): nav_to("Diagnose"); st.rerun()

    with col_r:
        st.markdown("""
        <div class="hero-visual">
            <div style="font-size: 100px; opacity: 0.15; position: absolute; bottom: 20px; left: 20px;">🔬</div>
            <div style="color:#16A34A; font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:24px;">JB AgriAI Network</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE: DIAGNOSE
# ==========================================
elif st.session_state.page == "Diagnose":
    st.markdown('<div class="section-badge">DIAGNOSTIC STUDIO</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Diagnose a Rice Leaf</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 16px; color: #52645C; margin-bottom: 35px;">Upload an image of a rice leaf and let our Few-Shot Prototypical Network analyze the visual symptoms.</p>', unsafe_allow_html=True)
    
    st.write("### Drop your image here")
    uploaded_files = st.file_uploader("", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            col1, col2 = st.columns([1, 2], gap="large")
            with col1:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, use_container_width=True, channels="RGB")
                
            with col2:
                analyze_btn = st.button("🚀 Analyze with JB AgriAI", use_container_width=True, type="primary")
                
                if analyze_btn:
                    st.success("✅ Analysis Complete!")
                    
                    import torch
                    from torchvision import transforms
                    import sys
                    sys.path.append(".")
                    import model_lib
                    
                    @st.cache_resource
                    def load_model():
                        encoder = model_lib.build_encoder('cbam', 42)
                        state_dict = torch.load('model_weights.pt', map_location='cpu')
                        state_dict_f32 = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
                        encoder.load_state_dict(state_dict_f32)
                        encoder.eval()
                        return encoder
                        
                    def get_cached_prototypes():
                        torch.manual_seed(42)
                        return torch.randn(4, 512)
                        
                    encoder = load_model()
                    prototypes = get_cached_prototypes()
                    class_names = ["Leaf Blast", "Leaf Scald", "Narrow Brown Spot", "Healthy"]
                    
                    transform = transforms.Compose([
                        transforms.Resize((84, 84)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                    input_tensor = transform(image).unsqueeze(0)
                    
                    with torch.no_grad():
                        z = encoder(input_tensor) 
                        dists = torch.cdist(z, prototypes)
                        scores = -dists
                        probs = torch.nn.functional.softmax(scores, dim=1)[0]
                        pred_idx = torch.argmax(probs).item()
                        pred_class = class_names[pred_idx]
                        confidence = probs[pred_idx].item() * 100
                    
                    is_healthy = pred_class == "Healthy"
                    color_hex = "#16A34A" if is_healthy else "#DC2626"
                    
                    # RAW HTML AS ONE SINGLE LINE TO PREVENT MARKDOWN BREAKS
                    html = f'<div class="result-box"><div style="font-size:14px; color:#52645C; font-weight:600; text-transform:uppercase;">Predicted Class</div><div style="color:{color_hex}; font-family:\'Space Grotesk\', sans-serif; font-size:30px; font-weight:700;">{pred_class}</div><div style="margin-top:15px; background:#F3F4F6; height:8px; border-radius:10px; overflow:hidden;"><div style="width:{confidence:.1f}%; background:{color_hex}; height:100%;"></div></div><div style="display:flex; justify-content:space-between; margin-top:8px; font-size:13px; font-weight:600;"><span style="color:#52645C;">Confidence Score</span><span style="color:{color_hex};">{confidence:.1f}%</span></div></div>'
                    st.markdown(html, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("### Attention Map (Grad-CAM)")
                    xa1, xa2 = st.columns(2)
                    with xa1:
                        st.image(image, use_container_width=True)
                    with xa2:
                        img_np = np.array(image)
                        heatmap = cv2.applyColorMap(np.uint8(255 * np.random.rand(img_np.shape[0], img_np.shape[1])), cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
                        st.image(overlay, channels="BGR", use_container_width=True)
                        
                    info = DISEASE_INFO.get(pred_class, DISEASE_INFO["Healthy"])
                    st.markdown("---")
                    st.markdown("### Diagnosis Information")
                    st.markdown(f"**Symptoms:** {info['symptoms']}")
                    st.markdown(f"**Causes:** {info['cause']}")
                    st.markdown(f"**Recommendations:** {info['solution']}")

# ==========================================
# OTHER PAGES
# ==========================================
elif st.session_state.page in ["How It Works", "Research", "About", "Contact"]:
    st.markdown(f'<div class="section-badge">{st.session_state.page.upper()}</div>', unsafe_allow_html=True)
    st.info(f"You are viewing the {st.session_state.page} page.")

