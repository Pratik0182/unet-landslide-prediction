import os
import h5py
import numpy as np
import torch
import streamlit as st
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import sys

# Add current path to sys for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.landslide_model import UNet
from src.landslide_main import tta_predict

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Landslide Prediction | Premium Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    /* Styling for a modern dashboard */
    .main {
        background-color: #0d1117;
    }
    .stSidebar {
        background-color: #161b22;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background-color: #21262d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .plot-container {
        border: 1px solid #30363d;
        border-radius: 15px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def load_trained_model(model_path, device):
    model = UNet(in_channels=14, out_channels=1).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    return None

def normalize_channels(image):
    # standard min-max for visualization
    m_min, m_max = image.min(), image.max()
    if m_max > m_min:
        return (image - m_min) / (m_max - m_min)
    return np.zeros_like(image)

def get_rgb_visual(h5_img):
    # Extract bands 3, 2, 1 (standard RGB from multispectral)
    # Most Landslide sets use specific orders, assuming common B3,B2,B1 for RGB
    rgb = h5_img[:, :, [3, 2, 1]]
    return normalize_channels(rgb)

def main():
    st.title("🛰️ Landslide Prediction Dashboard")
    st.subheader("Deep Learning (Attention U-Net) for Geospatial Landslide Prediction")
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("🔧 Model Configuration")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.info(f"Model Engine: **{str(device).upper()}**")
    
    model_file = st.sidebar.text_input("Model Weights Path", "models/unet_landslide.pth")
    model = load_trained_model(model_file, device)
    
    if not model:
        st.sidebar.error("❌ Model not found. Check the path!")
    else:
        st.sidebar.success("✅ Model Loaded Successfully")

    st.sidebar.markdown("---")
    st.sidebar.header("📁 Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Landslide H5 File", type=["h5"])
    
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    use_tta = st.sidebar.checkbox("Enable 8x TTA (Better accuracy)", value=True)
    
    if uploaded_file is not None:
        # Save temp 
        with open("temp_img.h5", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with h5py.File("temp_img.h5", "r") as f:
            image_data = f["img"][:]
            
        st.markdown("### 📊 Live Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        # --- PROCESSING ---
        with st.spinner("Analyzing satellite data..."):
            rgb_viz = get_rgb_visual(image_data)
            
            # Prediction
            input_tensor = torch.from_numpy(normalize_channels(image_data)).float().permute(2, 0, 1).unsqueeze(0)
            
            if use_tta:
                prediction = tta_predict(model, input_tensor, device)
            else:
                with torch.no_grad():
                    prediction = model(input_tensor.to(device)).cpu().squeeze().numpy()
            
            binary_mask = (prediction > confidence_threshold).astype(np.uint8)
            
        with col1:
            st.write("**True Color Image (RGB)**")
            st.image(rgb_viz, use_column_width=True)
            
        with col2:
            st.write("**Predicted Landslide Mask**")
            # Using heatmaps/plotly for premium look
            fig = px.imshow(prediction, color_continuous_scale='Viridis', labels=dict(color="Risk"))
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🧬 Detailed Overlay")
        
        # Transparent overlay for premium visualization
        overlay_opacity = st.slider("Overlay Transparency", 0.0, 1.0, 0.4)
        
        overlay = rgb_viz.copy()
        # Highlight mask in red
        red_channel = overlay[:, :, 0]
        red_channel[binary_mask == 1] = (1.0 * (1-overlay_opacity)) + (overlay[binary_mask == 1, 0] * overlay_opacity) 
        overlay[:, :, 0] = np.clip(red_channel, 0, 1)

        st.image(overlay, use_column_width=True, caption="Final Prediction Overlay Mapping")

        # Metrics display
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Detected Pixels", f"{np.sum(binary_mask)}")
        with m2:
            st.metric("Avg Risk Score", f"{np.mean(prediction):.4f}")
        with m3:
            st.metric("Max Confidence", f"{np.max(prediction):.2f}")
            
    else:
        # Default Welcome State
        st.info("👋 Upload a satellite image (.h5) in the sidebar to begin Landslide risk assessment.")
        st.markdown("""
        ### Features of this Model:
        *   **Attention Gates:** Automatically suppresses background and focuses on potential slide areas.
        *   **Multi-spectral Support:** Analyzes 14 parallel bands for signature identification.
        *   **TTA-Enabled:** Enhanced accuracy via Test-Time Augmentation over 8 axes.
        """)

if __name__ == "__main__":
    main()
