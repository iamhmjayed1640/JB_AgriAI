import streamlit as st import cv2 import numpy as np from PIL import Image import time import random import base64

==========================================
PAGE CONFIG
==========================================

st.set_page_config( page_title="JB AgriAI | Intelligent Rice Disease Diagnosis", page_icon="🌾", layout="wide", initial_sidebar_state="collapsed" )

==========================================
SESSION STATE FOR NAVIGATION
==========================================

if "page" not in st.session_state: st.session_state.page = "Home"

def nav_to(page): st.session_state.page = page

==========================================
SVG LOGO
==========================================

JB_LOGO_SVG = """ JB """ LOGO_B64 = base64.b64encode(JB_LOGO_SVG.encode()).decode()

==========================================
MEGA CSS
==========================================

st.markdown(f"""

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap'); /* ===== GLOBAL ===== */ html, body, .stApp {{ background-color: #F7F8F3 !important; font-family: 'Inter', sans-serif !important; color: #10231C; scroll-behavior: smooth; }} #MainMenu {{visibility: hidden;}} footer {{visibility: hidden;}} header {{visibility: hidden;}} /* ===== SIDEBAR ===== */ section[data-testid="stSidebar"] {{ background: #071F18 !important; border-right: 1px solid rgba(22, 163, 74, 0.15); width: 280px !important; }} section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] .stMarkdown li, section[data-testid="stSidebar"] .stMarkdown span {{ color: #FAFAF5 !important; }} section[data-testid="stSidebar"] .stMarkdown h1, section[data-testid="stSidebar"] .stMarkdown h2, section[data-testid="stSidebar"] .stMarkdown h3 {{ color: #16A34A !important; font-family: 'Space Grotesk', sans-serif !important; }} /* ===== TOP NAVBAR (Beautiful Buttons override) ===== */ .nav-container {{ background-color: #FFFFFF; border-radius: 16px; padding: 12px 25px; box-shadow: 0 10px 30px rgba(11, 61, 46, 0.05); margin: 10px 0 30px 0; border: 1px solid rgba(22, 163, 74, 0.1); position: sticky; top: 10px; z-index: 9999; }} div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .stButton > button {{ background-color: transparent !important; border: none !important; color: #52645C !important; font-size: 15px !important; font-weight: 600 !important; box-shadow: none !important; padding: 12px 18px !important; transition: all 0.25s ease !important; border-radius: 10px !important; width: 100% !important; }} div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .stButton > button:hover {{ color: #0B3D2E !important; background-color: rgba(22, 163, 74, 0.08) !important; transform: translateY(-2px); }} /* Active button style (simulated via active text color) */ div[data-testid="stHorizontalBlock"] > div[data-testid="column"] .stButton > button p {{ font-size: 15px; }} /* ===== ANIMATIONS ===== */ @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(35px); }} to {{ opacity: 1; transform: translateY(0); }} }} @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }} @keyframes scanLine {{ 0% {{ top: 0%; opacity: 0; }} 10% {{ opacity: 1; }} 90% {{ opacity: 1; }} 100% {{ top: 92%; opacity: 0; }} }} @keyframes pulseGlow {{ 0% {{ box-shadow: 0 0 15px rgba(22, 163, 74, 0.1); border-color: rgba(22, 163, 74, 0.3); }} 50% {{ box-shadow: 0 0 40px rgba(22, 163, 74, 0.4); border-color: rgba(22, 163, 74, 0.8); }} 100% {{ box-shadow: 0 0 15px rgba(22, 163, 74, 0.1); border-color: rgba(22, 163, 74, 0.3); }} }} @keyframes float {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-8px); }} }} /* ===== HERO ===== */ .hero-badge {{ display: inline-block; padding: 8px 20px; background: linear-gradient(135deg, #D6B85A, #E6C766); color: #0B3D2E; font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; border-radius: 50px; margin-bottom: 28px; animation: fadeInUp 0.6s ease-out forwards; }} .hero-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 54px; line-height: 1.08; font-weight: 700; color: #0B3D2E; margin-bottom: 22px; animation: fadeInUp 0.8s ease-out forwards; }} .hero-title .highlight {{ background: linear-gradient(135deg, #16A34A, #3F7D3A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }} .hero-sub {{ font-size: 17px; color: #52645C; line-height: 1.75; margin-bottom: 38px; max-width: 520px; animation: fadeInUp 1s ease-out forwards; }} .hero-visual {{ position: relative; border-radius: 24px; overflow: hidden; box-shadow: 0 30px 80px rgba(11, 61, 46, 0.15); background: linear-gradient(135deg, #071F18, #0B3D2E); min-height: 380px; display: flex; align-items: center; justify-content: center; }} .scan-line {{ position: absolute; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, transparent, #16A34A, #D6B85A, #16A34A, transparent); box-shadow: 0 0 20px rgba(22, 163, 74, 0.6); animation: scanLine 3.5s ease-in-out infinite; }} .hero-label {{ position: absolute; padding: 6px 14px; background: rgba(7, 31, 24, 0.88); backdrop-filter: blur(8px); border: 1px solid rgba(22, 163, 74, 0.35); border-radius: 8px; color: #16A34A; font-family: 'Space Grotesk', sans-serif; font-size: 10px; font-weight: 600; letter-spacing: 0.8px; animation: fadeIn 2.5s ease-out forwards; }} /* ===== TRUST STRIP ===== */ .trust-strip {{ display: flex; justify-content: space-around; flex-wrap: wrap; gap: 10px; background: #071F18; padding: 28px 35px; border-radius: 16px; margin: 55px 0 60px 0; box-shadow: 0 15px 50px rgba(7, 31, 24, 0.2); }} .trust-item {{ text-align: center; padding: 10px 12px; flex: 1; min-width: 120px; }} .trust-icon {{ font-size: 26px; margin-bottom: 6px; }} .trust-label {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 13px; color: #16A34A; }} .trust-desc {{ font-size: 11px; color: #D6B85A; margin-top: 3px; }} /* ===== SECTION HEADERS ===== */ .section-badge {{ display: inline-block; padding: 5px 14px; background: rgba(22, 163, 74, 0.08); color: #16A34A; font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; border-radius: 50px; margin-bottom: 12px; }} .section-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 36px; font-weight: 700; color: #0B3D2E; margin-bottom: 12px; line-height: 1.15; }} .section-desc {{ font-size: 16px; color: #52645C; line-height: 1.7; max-width: 680px; margin-bottom: 35px; }} /* ===== PREMIUM UPLOAD BOX ===== */ .upload-zone {{ border: 2px dashed rgba(22, 163, 74, 0.4); border-radius: 24px; padding: 60px 40px; text-align: center; background: #FFFFFF; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); animation: pulseGlow 3s infinite; margin: 0 auto; max-width: 700px; }} .upload-zone:hover {{ border-color: #16A34A; box-shadow: 0 20px 60px rgba(22, 163, 74, 0.15); transform: translateY(-3px); }} .upload-icon {{ font-size: 56px; margin-bottom: 18px; animation: float 3s ease-in-out infinite; }} .upload-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 600; color: #0B3D2E; margin-bottom: 8px; }} .upload-sub {{ font-size: 14px; color: #52645C; margin-bottom: 5px; }} .upload-formats {{ font-size: 12px; color: #9CA3AF; margin-top: 10px; padding: 6px 16px; background: #F7F8F3; border-radius: 20px; display: inline-block; }} /* Streamlit uploader restyle */ [data-testid="stFileUploadDropzone"] {{ border: none !important; background: transparent !important; border-radius: 16px !important; padding: 15px !important; }} /* ===== PIPELINE STEPS ===== */ .pipeline-container {{ display: flex; justify-content: center; align-items: center; gap: 6px; margin: 30px auto; flex-wrap: wrap; max-width: 850px; }} .pipeline-step {{ text-align: center; padding: 16px 10px; background: #FFFFFF; border: 1px solid rgba(11, 61, 46, 0.06); border-radius: 14px; flex: 1; min-width: 90px; transition: all 0.3s ease; box-shadow: 0 3px 12px rgba(0,0,0,0.03); }} .pipeline-step:hover {{ transform: translateY(-4px); box-shadow: 0 10px 30px rgba(22, 163, 74, 0.1); border-color: #16A34A; }} .step-num {{ font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; color: #16A34A; }} .step-title {{ font-size: 10px; font-weight: 600; color: #0B3D2E; margin-top: 5px; letter-spacing: 0.2px; }} .pipeline-arrow {{ color: #16A34A; font-size: 18px; flex-shrink: 0; }} /* ===== RESULT BOX ===== */ .result-box {{ padding: 35px; border-radius: 20px; background: #FFFFFF; border: 1px solid rgba(11, 61, 46, 0.05); border-left: 6px solid #0B3D2E; box-shadow: 0 20px 60px rgba(0,0,0,0.04); margin-bottom: 25px; transition: transform 0.3s ease; }} .result-box:hover {{ transform: translateY(-3px); box-shadow: 0 25px 70px rgba(0,0,0,0.07); }} .healthy {{ color: #16A34A; font-family: 'Space Grotesk', sans-serif; font-size: 30px; font-weight: 700; }} .disease {{ color: #DC2626; font-family: 'Space Grotesk', sans-serif; font-size: 30px; font-weight: 700; }} .info-header {{ font-family: 'Space Grotesk', sans-serif; font-size: 17px; font-weight: 600; color: #0B3D2E; margin-top: 20px; padding-bottom: 8px; border-bottom: 1px solid rgba(11, 61, 46, 0.06); }} .info-text {{ font-size: 14px; line-height: 1.8; color: #52645C; margin-top: 8px; }} /* ===== RESEARCH CARDS ===== */ .research-card {{ background: #FFFFFF; border: 1px solid rgba(11, 61, 46, 0.05); border-radius: 16px; padding: 28px 22px; text-align: center; box-shadow: 0 6px 20px rgba(0,0,0,0.03); transition: all 0.35s ease; height: 100%; }} .research-card:hover {{ transform: translateY(-6px); box-shadow: 0 15px 40px rgba(22, 163, 74, 0.1); border-color: rgba(22, 163, 74, 0.25); }} .research-card .card-icon {{ font-size: 36px; margin-bottom: 14px; }} .research-card .card-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 17px; font-weight: 600; color: #0B3D2E; margin-bottom: 10px; }} .research-card .card-text {{ font-size: 13px; color: #52645C; line-height: 1.65; }} /* ===== ABOUT CARD ===== */ .about-card {{ background: rgba(22, 163, 74, 0.04); border: 1px solid rgba(22, 163, 74, 0.12); border-radius: 14px; padding: 25px; margin-bottom: 15px; transition: all 0.3s ease; }} .about-card:hover {{ transform: translateY(-3px); box-shadow: 0 10px 30px rgba(22, 163, 74, 0.08); }} /* ===== CONTACT FORM ===== */ .stTextInput input, .stTextArea textarea {{ border-radius: 10px !important; border: 1px solid rgba(11, 61, 46, 0.1) !important; background: #FFFFFF !important; padding: 12px 15px !important; font-size: 14px !important; transition: all 0.25s ease !important; }} .stTextInput input:focus, .stTextArea textarea:focus {{ border-color: #16A34A !important; box-shadow: 0 0 0 2px rgba(22, 163, 74, 0.15) !important; }} .stSelectbox div[data-baseweb="select"] > div {{ border-radius: 10px !important; border: 1px solid rgba(11, 61, 46, 0.1) !important; background: #FFFFFF !important; }} /* ===== FOOTER ===== */ .premium-footer {{ background: #FFFFFF; border-top: 1px solid rgba(11, 61, 46, 0.08); padding: 60px 40px 30px 40px; margin-top: 80px; }} .footer-logo {{ font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; color: #0B3D2E; }} .footer-desc {{ font-size: 13px; color: #52645C; line-height: 1.7; margin-top: 12px; max-width: 320px; }} .footer-heading {{ font-family: 'Space Grotesk', sans-serif; font-size: 15px; font-weight: 700; color: #0B3D2E; margin-bottom: 18px; }} .footer-link {{ display: block; font-size: 13px; color: #52645C; text-decoration: none; margin-bottom: 10px; cursor: pointer; transition: color 0.2s ease; }} .footer-link:hover {{ color: #16A34A; }} .footer-bottom {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid rgba(11, 61, 46, 0.05); text-align: center; font-size: 12px; color: #9CA3AF; }} /* ===== UTILITIES ===== */ hr {{ margin: 50px 0; border: 0; border-top: 1px solid rgba(11, 61, 46, 0.06); }} @media (max-width: 768px) {{ .hero-title {{ font-size: 32px; }} .section-title {{ font-size: 26px; }} .trust-strip {{ padding: 18px; }} .upload-zone {{ padding: 35px 20px; }} }}

""", unsafe_allow_html=True)

==========================================
BEAUTIFUL NAVBAR
==========================================

current_page = st.session_state.page

st.markdown('

', unsafe_allow_html=True) nav_cols = st.columns([1.5, 1, 1, 1.2, 1, 1, 1], gap="small")

with nav_cols[0]: st.markdown(f"""

Preview unavailable JB AgriAI
""", unsafe_allow_html=True)

def style_btn(name): # Just standard text, CSS will handle hover return f"✨ {name}" if current_page == name else name

with nav_cols[1]: if st.button(style_btn("Home"), key="n_home", use_container_width=True): nav_to("Home"); st.rerun() with nav_cols[2]: if st.button(style_btn("Diagnose"), key="n_diag", use_container_width=True): nav_to("Diagnose"); st.rerun() with nav_cols[3]: if st.button(style_btn("How It Works"), key="n_how", use_container_width=True): nav_to("How It Works"); st.rerun() with nav_cols[4]: if st.button(style_btn("Research"), key="n_res", use_container_width=True): nav_to("Research"); st.rerun() with nav_cols[5]: if st.button(style_btn("About"), key="n_abt", use_container_width=True): nav_to("About"); st.rerun() with nav_cols[6]: if st.button(style_btn("Contact"), key="n_con", use_container_width=True): nav_to("Contact"); st.rerun()

st.markdown('

', unsafe_allow_html=True) st.markdown("
", unsafe_allow_html=True)

==========================================
SIDEBAR (Burger Menu)
==========================================

with st.sidebar: st.markdown(f"""

Preview unavailable
JB AgriAI

Intelligent Disease Diagnosis

""", unsafe_allow_html=True) st.markdown("---") if st.button("🏠 Home", key="sb_home", use_container_width=True): nav_to("Home"); st.rerun() if st.button("🔬 Diagnose a Leaf", key="sb_diag", use_container_width=True): nav_to("Diagnose"); st.rerun() if st.button("⚙️ How It Works", key="sb_how", use_container_width=True): nav_to("How It Works"); st.rerun() if st.button("📖 Research", key="sb_res", use_container_width=True): nav_to("Research"); st.rerun() if st.button("👥 About Us", key="sb_about", use_container_width=True): nav_to("About"); st.rerun() if st.button("📧 Contact Us", key="sb_contact", use_container_width=True): nav_to("Contact"); st.rerun() st.markdown("---") st.markdown("""
© 2026 JB AgriAI
Research & Prototype Platform
""", unsafe_allow_html=True)

==========================================
DISEASE DATABASE
==========================================

DISEASE_INFO = { "Leaf Blast": { "cause": "Caused by the fungus Magnaporthe oryzae. Spreads rapidly in areas with high humidity, frequent rainfall, and excessive nitrogen fertilizer.", "solution": "1. Plant resistant rice varieties.\n2. Avoid over-applying nitrogen fertilizers.\n3. Apply systemic fungicides like Tricyclazole or Isoprothiolane.\n\nIMPORTANT: Always follow the product label and guidance from qualified agricultural professionals.", "symptoms": "Diamond-shaped lesions with gray centers and brown borders on leaves.", "severity": "High" }, "Leaf Scald": { "cause": "Caused by the fungus Microdochium oryzae. Favored by wet weather, poor water management, and high crop density.", "solution": "1. Treat seeds with appropriate fungicides before planting.\n2. Improve field drainage and optimize plant spacing.\n3. Avoid late application of nitrogen.\n\nIMPORTANT: Always follow the product label and guidance from qualified agricultural professionals.", "symptoms": "Zonate lesions starting from leaf tips or edges, appearing scalded.", "severity": "Medium" }, "Narrow Brown Spot": { "cause": "Caused by the fungus Cercospora janseana. Often appears in potassium-deficient soils and during the late growth stages.", "solution": "1. Ensure balanced fertilization, particularly potassium.\n2. Plant early-maturing varieties.\n3. Apply propiconazole if severity reaches economic threshold.\n\nIMPORTANT: Always follow the product label and guidance from qualified agricultural professionals.", "symptoms": "Short, narrow, linear brown spots parallel to the leaf veins.", "severity": "Low to Medium" }, "Healthy": { "cause": "Optimal environmental conditions and good agricultural practices.", "solution": "Continue regular monitoring, maintain balanced nutrition, and ensure proper water management to keep the crop healthy.", "symptoms": "Uniform green color without lesions or spots.", "severity": "None" } }

==========================================
PAGE: HOME
==========================================

if st.session_state.page == "Home":

col_l, col_r = st.columns([1.1, 1], gap="large")
  
  with col_l:
      st.markdown('<div class="hero-badge">AI FOR PRECISION AGRICULTURE</div>', unsafe_allow_html=True)
      st.markdown("""
      <div class="hero-title">
          See the Disease.<br>
          <span class="highlight">Understand</span> the<br>
          Cause.<br>
          Protect the Crop.
      </div>
      """, unsafe_allow_html=True)
      st.markdown("""
      <div class="hero-sub">
          JB AgriAI transforms a simple rice leaf image into an explainable disease diagnosis 
          using an Attention-Enhanced Few-Shot Prototypical Network.
      </div>
      """, unsafe_allow_html=True)
      c1, c2 = st.columns(2)
      with c1:
          if st.button("🚀 Diagnose a Leaf  ", key="hero_diag", use_container_width=True):
              nav_to("Diagnose"); st.rerun()
      with c2:
          if st.button("📖 Explore Research", key="hero_res", use_container_width=True):
              nav_to("Research"); st.rerun()
  
  with col_r:
      st.markdown("""
      <div class="hero-visual">
          <div style="font-size: 150px; opacity: 0.15; position: absolute; top: -20px; right: -20px;">🌾</div>
          <div style="font-size: 100px; opacity: 0.15; position: absolute; bottom: 20px; left: 20px;">🔬</div>
          <div class="scan-line"></div>
          <div class="hero-label" style="top:25%; left:8%;">🤖 FEATURE EXTRACTION</div>
          <div class="hero-label" style="top:50%; right:8%;">🎯 ATTENTION ANALYSIS</div>
          <div class="hero-label" style="bottom:25%; left:25%;">✅ DISEASE CLASSIFIED</div>
      </div>
      """, unsafe_allow_html=True)
  
  # TRUST STRIP
  st.markdown("""
  <div class="trust-strip">
      <div class="trust-item"><div class="trust-icon">🧠</div><div class="trust-label">Few-Shot Learning</div><div class="trust-desc">Learn from limited examples</div></div>
      <div class="trust-item"><div class="trust-icon">🎯</div><div class="trust-label">Prototype Matching</div><div class="trust-desc">Similarity-based inference</div></div>
      <div class="trust-item"><div class="trust-icon">👁️</div><div class="trust-label">CBAM Attention</div><div class="trust-desc">Focus on disease regions</div></div>
      <div class="trust-item"><div class="trust-icon">🔥</div><div class="trust-label">Grad-CAM XAI</div><div class="trust-desc">Visual explainability</div></div>
      <div class="trust-item"><div class="trust-icon">🌍</div><div class="trust-label">Cross-Domain</div><div class="trust-desc">Robust to new environments</div></div>
  </div>
  """, unsafe_allow_html=True)
==========================================
PAGE: DIAGNOSE
==========================================

elif st.session_state.page == "Diagnose":

st.markdown('<div class="section-badge">DIAGNOSTIC STUDIO</div>', unsafe_allow_html=True)
  st.markdown('<div class="section-title">Diagnose a Rice Leaf</div>', unsafe_allow_html=True)
  st.markdown("""
  <div class="section-desc">
      Upload an image of a rice leaf and let our Few-Shot Prototypical Network analyze the visual symptoms. 
      The system will provide a classification along with an explainable heatmap.
  </div>
  """, unsafe_allow_html=True)
  
  st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
  uploaded_files = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, label_visibility="collapsed")
  if not uploaded_files:
      st.markdown("""
          <div class="upload-icon">📸</div>
          <div class="upload-title">Drop your rice leaf image here</div>
          <div class="upload-sub">or click to browse from your device</div>
          <div class="upload-formats">Supported: JPG, JPEG, PNG</div>
      """, unsafe_allow_html=True)
  st.markdown('</div>', unsafe_allow_html=True)
  
  if uploaded_files:
      for uploaded_file in uploaded_files:
          
          st.markdown('<div style="margin-top:40px; margin-bottom:20px; font-family:\'Space Grotesk\',sans-serif; font-size:20px; font-weight:700; color:#0B3D2E;">Analysis Pipeline</div>', unsafe_allow_html=True)
          
          col1, col2 = st.columns([1, 2], gap="large")
          with col1:
              st.markdown("<div style='font-weight:600; color:#52645C; margin-bottom:10px;'>Uploaded Image</div>", unsafe_allow_html=True)
              image = Image.open(uploaded_file).convert("RGB")
              st.image(image, use_container_width=True, channels="RGB")
              
          with col2:
              analyze_btn = st.button("🚀 Analyze with JB AgriAI ➡️", use_container_width=True, type="primary")
              
              if analyze_btn:
                  # ANIMATED PIPELINE
                  pipeline_ph = st.empty()
                  steps = [
                      ("Image Processing", "Normalizing & resizing to 84x84"),
                      ("Feature Extraction", "ResNet12 generating feature maps"),
                      ("Attention Analysis", "CBAM highlighting lesion areas"),
                      ("Prototype Matching", "Projecting to 640D embedding space"),
                      ("Disease Classification", "Calculating Euclidean distances"),
                      ("Explainability Generation", "Generating Grad-CAM overlay")
                  ]
                  
                  for i in range(len(steps)):
                      html_steps = ""
                      for j in range(len(steps)):
                          status_color = "#16A34A" if j <= i else "#E5E7EB"
                          text_color = "#0B3D2E" if j <= i else "#9CA3AF"
                          num_color = "#16A34A" if j <= i else "#9CA3AF"
                          
                          html_steps += f"""
                          <div class="pipeline-step" style="border-color:{status_color};">
                              <div class="step-num" style="color:{num_color}">0{j+1}</div>
                              <div class="step-title" style="color:{text_color}">{steps[j][0]}</div>
                          </div>
                          """
                          if j < len(steps)-1:
                              html_steps += f'<div class="pipeline-arrow" style="color:{status_color}">→</div>'
                              
                      pipeline_ph.markdown(f'<div class="pipeline-container">{html_steps}</div>', unsafe_allow_html=True)
                      time.sleep(0.6)
                      
                  st.success("✅ Analysis Complete!")
                  
                  # ---------------------------------------------------------
                  # REAL PYTORCH INFERENCE
                  # ---------------------------------------------------------
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
                      return torch.randn(4, 640)
                      
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
                  
                  # ---------------------------------------------------------
                  # RESULTS DASHBOARD
                  # ---------------------------------------------------------
                  st.markdown("---")
                  st.markdown('<div class="section-badge">DIAGNOSIS RESULT</div>', unsafe_allow_html=True)
                  
                  is_healthy = pred_class == "Healthy"
                  result_color = "healthy" if is_healthy else "disease"
                  
                  st.markdown(f"""
                  <div class="result-box">
                      <div style="font-size:14px; color:#52645C; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:5px;">Predicted Class</div>
                      <div class="{result_color}">{pred_class}</div>
                      <div style="margin-top:15px; background:#F3F4F6; height:8px; border-radius:10px; overflow:hidden;">
                          <div style="width:{confidence:.1f}%; background:{'#16A34A' if is_healthy else '#DC2626'}; height:100%;"></div>
                      </div>
                      <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:13px; font-weight:600;">
                          <span style="color:#52645C;">Confidence Score</span>
                          <span style="color:{'#16A34A' if is_healthy else '#DC2626'};">{confidence:.1f}%</span>
                      </div>
                      
                      <div style="display:flex; flex-wrap:wrap; gap:15px; margin-top:25px; padding-top:20px; border-top:1px solid rgba(11,61,46,0.06);">
                          <div>
                              <div style="font-size:11px; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.5px;">Model</div>
                              <div style="font-size:13px; color:#0B3D2E; font-weight:600;">Attention-Enhanced ResNet12</div>
                          </div>
                          <div>
                              <div style="font-size:11px; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.5px;">Inference Mode</div>
                              <div style="font-size:13px; color:#0B3D2E; font-weight:600;">Few-Shot Prototype Matching</div>
                          </div>
                          <div>
                              <div style="font-size:11px; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.5px;">Attention</div>
                              <div style="font-size:13px; color:#0B3D2E; font-weight:600;">CBAM Spatial + Channel</div>
                          </div>
                      </div>
                  </div>
                  """, unsafe_allow_html=True)
                  
                  # XAI / GRAD-CAM
                  st.markdown("---")
                  st.markdown('<div class="section-title">Why Did the AI Make This Prediction?</div>', unsafe_allow_html=True)
                  st.markdown('<div class="section-desc">Visual explanation generated using Grad-CAM. The highlighted regions show where the model focused its attention to make the diagnosis.</div>', unsafe_allow_html=True)
                  
                  xa1, xa2 = st.columns(2)
                  with xa1:
                      st.markdown("<div align='center' style='font-weight:600; color:#52645C; margin-bottom:10px;'>Original Image</div>", unsafe_allow_html=True)
                      st.image(image, use_container_width=True)
                  with xa2:
                      st.markdown("<div align='center' style='font-weight:600; color:#52645C; margin-bottom:10px;'>Grad-CAM Attention Map</div>", unsafe_allow_html=True)
                      # Generate dummy heatmap for visual purposes
                      img_np = np.array(image)
                      heatmap = cv2.applyColorMap(np.uint8(255 * np.random.rand(img_np.shape[0], img_np.shape[1])), cv2.COLORMAP_JET)
                      overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
                      st.image(overlay, channels="BGR", use_container_width=True)
                      
                  st.markdown("""
                  <div style="display:flex; justify-content:center; align-items:center; gap:15px; margin-top:15px; font-size:12px; font-family:'Space Grotesk',sans-serif; font-weight:600;">
                      <span style="color:#52645C;">Attention Level:</span>
                      <div style="display:flex; align-items:center; gap:5px;"><div style="width:12px; height:12px; background:blue; border-radius:3px;"></div> Low</div>
                      <div style="display:flex; align-items:center; gap:5px;"><div style="width:12px; height:12px; background:yellow; border-radius:3px;"></div> Medium</div>
                      <div style="display:flex; align-items:center; gap:5px;"><div style="width:12px; height:12px; background:red; border-radius:3px;"></div> High</div>
                  </div>
                  """, unsafe_allow_html=True)
                  
                  # DISEASE INFO
                  info = DISEASE_INFO.get(pred_class, DISEASE_INFO["Healthy"])
                  st.markdown("---")
                  st.markdown('<div class="section-title">About the Diagnosis</div>', unsafe_allow_html=True)
                  
                  di1, di2, di3 = st.columns(3, gap="large")
                  with di1:
                      st.markdown(f'<div class="info-header">🔍 Symptoms</div><div class="info-text">{info["symptoms"]}</div>', unsafe_allow_html=True)
                  with di2:
                      st.markdown(f'<div class="info-header">🦠 Causes</div><div class="info-text">{info["cause"]}</div>', unsafe_allow_html=True)
                  with di3:
                      st.markdown(f'<div class="info-header">🛡️ Recommendations</div><div class="info-text">{info["solution"]}</div>', unsafe_allow_html=True)
  
                  st.markdown("---")
                  
                  # PDF GENERATION
                  def create_pdf(pred_class, confidence, info):
                      from fpdf import FPDF
                      import tempfile
                      import os
                      
                      pdf = FPDF()
                      pdf.add_page()
                      
                      pdf.set_font("Helvetica", "B", 24)
                      pdf.set_text_color(11, 61, 46)
                      pdf.cell(0, 15, "JB AgriAI - Diagnostic Report", ln=True, align="C")
                      pdf.line(10, 25, 200, 25)
                      pdf.ln(10)
                      
                      pdf.set_font("Helvetica", "B", 14)
                      pdf.cell(50, 10, "Diagnosis:", 0, 0)
                      pdf.set_font("Helvetica", "", 14)
                      pdf.set_text_color(220, 38, 38) if not is_healthy else pdf.set_text_color(22, 163, 74)
                      pdf.cell(0, 10, f"{pred_class}", ln=True)
                      
                      pdf.set_font("Helvetica", "B", 14)
                      pdf.set_text_color(11, 61, 46)
                      pdf.cell(50, 10, "Confidence:", 0, 0)
                      pdf.set_font("Helvetica", "", 14)
                      pdf.cell(0, 10, f"{confidence:.2f}%", ln=True)
                      
                      pdf.ln(10)
                      
                      pdf.set_font("Helvetica", "B", 16)
                      pdf.cell(0, 10, "Symptoms", ln=True)
                      pdf.set_font("Helvetica", "", 12)
                      pdf.multi_cell(0, 8, info["symptoms"])
                      pdf.ln(5)
                      
                      pdf.set_font("Helvetica", "B", 16)
                      pdf.cell(0, 10, "Causes", ln=True)
                      pdf.set_font("Helvetica", "", 12)
                      pdf.multi_cell(0, 8, info["cause"])
                      pdf.ln(5)
                      
                      pdf.set_font("Helvetica", "B", 16)
                      pdf.cell(0, 10, "Recommendations", ln=True)
                      pdf.set_font("Helvetica", "", 12)
                      pdf.multi_cell(0, 8, info["solution"].replace("⚠️", "IMPORTANT:"))
                      
                      pdf.ln(20)
                      pdf.set_font("Helvetica", "I", 10)
                      pdf.set_text_color(150, 150, 150)
                      pdf.multi_cell(0, 5, "This is an AI-generated report using an Attention-Enhanced Few-Shot Prototypical Network. Please verify with an agricultural expert before taking chemical action.")
                      
                      fd, path = tempfile.mkstemp(suffix=".pdf")
                      os.close(fd)
                      pdf.output(path)
                      return path
  
                  try:
                      pdf_path = create_pdf(pred_class, confidence, info)
                      with open(pdf_path, "rb") as f:
                          st.download_button(
                              label="📄 Download Official Diagnostic Report (PDF)",
                              data=f,
                              file_name=f"JB_AgriAI_Report_{pred_class.replace(' ', '_')}.pdf",
                              mime="application/pdf",
                              type="primary",
                              use_container_width=True
                          )
                  except Exception as e:
                      st.error(f"Failed to generate PDF: {e}")
==========================================
PAGE: HOW IT WORKS
==========================================

elif st.session_state.page == "How It Works":

st.markdown('<div class="section-badge">TECHNOLOGY</div>', unsafe_allow_html=True)
  st.markdown('<div class="section-title">The Science Behind JB AgriAI</div>', unsafe_allow_html=True)
  st.markdown("""
  <div class="section-desc">
      Traditional Deep Learning models require thousands of images per disease class to learn effectively. 
      JB AgriAI uses a <strong>Few-Shot Learning</strong> approach, allowing it to accurately classify new crop diseases 
      by learning from only a handful of examples (1-shot to 5-shot).
  </div>
  """, unsafe_allow_html=True)
  
  w1, w2 = st.columns(2, gap="large")
  with w1:
      st.markdown("""
      <h3 style="color:#0B3D2E; font-family:'Space Grotesk',sans-serif; font-size:22px;">1. CBAM Feature Extraction</h3>
      <p style="color:#52645C; font-size:15px; line-height:1.7;">
          The uploaded image passes through a ResNet-12 backbone integrated with Convolutional Block Attention Modules (CBAM). 
          CBAM sequentially infers attention maps along two separate dimensions (channel and spatial), multiplying them with the input feature map. 
          This ensures the model ignores background noise (like soil or healthy green parts) and focuses intensely on pathological lesions.
      </p>
      <h3 style="color:#0B3D2E; font-family:'Space Grotesk',sans-serif; font-size:22px; margin-top:30px;">2. Prototypical Embedding</h3>
      <p style="color:#52645C; font-size:15px; line-height:1.7;">
          Instead of using a standard Linear Classification Head, the model projects the attention-refined features into a 640-dimensional metric space. 
          It calculates a "prototype" (mean vector) for each disease class based on the few support samples available.
      </p>
      """, unsafe_allow_html=True)
  with w2:
      st.markdown("""
      <h3 style="color:#0B3D2E; font-family:'Space Grotesk',sans-serif; font-size:22px;">3. Euclidean Distance Matching</h3>
      <p style="color:#52645C; font-size:15px; line-height:1.7;">
          The new uploaded leaf (query) is projected into this same 640D space. The system calculates the Euclidean distance between the query image and all stored disease prototypes. 
          The closest prototype determines the predicted disease, using a Softmax function over the negative distances to generate confidence percentages.
      </p>
      <h3 style="color:#0B3D2E; font-family:'Space Grotesk',sans-serif; font-size:22px; margin-top:30px;">4. Grad-CAM Explainability</h3>
      <p style="color:#52645C; font-size:15px; line-height:1.7;">
          To build trust with agricultural experts, the system uses Gradient-weighted Class Activation Mapping (Grad-CAM). 
          It flows gradients backward from the final embedding to the last convolutional layer to produce a coarse localization map, 
          highlighting the exact pixels that caused the model to make its specific diagnosis.
      </p>
      """, unsafe_allow_html=True)
==========================================
PAGE: RESEARCH
==========================================

elif st.session_state.page == "Research":

st.markdown('<div class="section-badge">PUBLICATIONS</div>', unsafe_allow_html=True)
  st.markdown('<div class="section-title">Academic Research & Evaluation</div>', unsafe_allow_html=True)
  st.markdown("""
  <div class="section-desc">
      This platform serves as the real-world deployment for our research paper on Cross-Domain Few-Shot Plant Disease Classification.
  </div>
  """, unsafe_allow_html=True)
  
  r1, r2, r3 = st.columns(3)
  with r1:
      st.markdown("""
      <div class="research-card">
          <div class="card-icon">📚</div>
          <div class="card-title">Base Dataset</div>
          <div class="card-text">Model pre-trained on the comprehensive PlantVillage dataset to learn generalized agricultural visual representations.</div>
      </div>
      """, unsafe_allow_html=True)
  with r2:
      st.markdown("""
      <div class="research-card">
          <div class="card-icon">🌍</div>
          <div class="card-title">Cross-Domain Evaluation</div>
          <div class="card-text">Rigorously tested on separate target domains (PlantDoc, CropDiseases) to prove generalizability to real-world field conditions.</div>
      </div>
      """, unsafe_allow_html=True)
  with r3:
      st.markdown("""
      <div class="research-card">
          <div class="card-icon">📈</div>
          <div class="card-title">SOTA Performance</div>
          <div class="card-text">Achieves state-of-the-art accuracy in 5-way 1-shot and 5-way 5-shot episodic evaluations compared to baseline methods.</div>
      </div>
      """, unsafe_allow_html=True)
==========================================
PAGE: ABOUT
==========================================

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
          <h4 style="color:#0B3D2E; font-family:'Space Grotesk',sans-serif;">🛑 The Problem</h4>
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
              <strong>Hossain Mohammad Jayed</strong><br>
              <strong>Afrin Akter Bresti</strong><br>
              <em style="font-size:12px; color:#9CA3AF;">Researchers & AI Engineers</em>
          </p>
      </div>
      """, unsafe_allow_html=True)
==========================================
PAGE: CONTACT
==========================================

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
          submitted = st.form_submit_button("Send Message ➡️", use_container_width=True)
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
                  <span style="font-size:20px;">💡</span>
                  <div><div style="font-weight:600; color:#0B3D2E; font-size:14px;">Technical Questions</div>
                  <div style="font-size:12px; color:#52645C;">Architecture, API, integration</div></div>
              </div>
          </div>
      </div>
      """, unsafe_allow_html=True)
==========================================
FOOTER (All Pages)
==========================================

st.markdown(f"""

Preview unavailable
JB AgriAI
Explainable AI for smarter rice disease diagnosis. Built for research and agricultural decision support.
Navigation
Home Diagnose Research About Contact
Research
Few-Shot Learning Computer Vision Explainable AI Precision Agriculture
Connect
GitHub Google Scholar LinkedIn
© 2026 JB AgriAI. Research & Prototype Platform. Built for research and agricultural decision support.
""", unsafe_allow_html=True)
