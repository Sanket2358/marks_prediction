import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# 1. Page Configuration
st.set_page_config(
    page_title="AI Predictor Pro",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. CSS with Floating Title & Static Glow Output
st.markdown("""
    <style>
    /* Define Adaptive Colors for Light/Dark Mode */
    :root {
        --title-gradient: linear-gradient(45deg, #1e3a8a, #9333ea, #1e3a8a);
        --glow-shadow: rgba(147, 51, 234, 0.3);
        --metric-glow: rgba(147, 51, 234, 0.5);
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --title-gradient: linear-gradient(45deg, #00f6ff, #ff007f, #00f6ff);
            --glow-shadow: rgba(0, 246, 255, 0.4);
            --metric-glow: rgba(0, 246, 255, 0.6);
        }
    }

    /* Subtle Background Animation */
    .stApp {
        background: linear-gradient(-45deg, var(--background-color), var(--secondary-background-color), var(--background-color));
        background-size: 300% 300%;
        animation: bg-shift 15s ease infinite;
    }
    
    @keyframes bg-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Floating & Glowing Title Animation */
    .animated-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        font-family: 'Segoe UI', system-ui, sans-serif;
        background: var(--title-gradient);
        background-size: 200% auto;
        color: #ffffff;
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: 
            gradient-move 3s linear infinite, 
            float-up-down 4s ease-in-out infinite;
        text-shadow: 0px 5px 20px var(--glow-shadow);
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        padding-top: 10px; 
    }

    @keyframes gradient-move {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    @keyframes float-up-down {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    /* Smooth Fade-In Subtext */
    .fade-in-text {
        text-align: center;
        color: var(--text-color);
        opacity: 0; 
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 2.5rem;
        animation: fadeInUp 1s ease-out 0.3s forwards;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 0.8; transform: translateY(0); }
    }

    /* --- Output Metric Styling --- */
    
    /* 1. Slide-up effect for the whole metric card when it appears */
    div[data-testid="metric-container"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 15px;
        padding: 25px;
        animation: slide-up-fade 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        transition: all 0.4s ease;
    }
    
    @keyframes slide-up-fade {
        from { opacity: 0; transform: translateY(30px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 10px 25px var(--glow-shadow);
        border: 1px solid var(--primary-color);
    }

    /* 2. Static Neon Glow effect specifically on the output NUMBER (e.g., 47.32) */
    div[data-testid="stMetricValue"] > div {
        font-weight: 800;
        color: var(--primary-color);
        text-shadow: 0 0 15px var(--metric-glow), 0 0 30px var(--metric-glow);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load the Model
@st.cache_resource
def load_model():
    """Loads the pickled KNeighborsRegressor model."""
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# 4. Animated Headers
st.markdown('<div class="animated-title">AI Performance Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="fade-in-text">Powered by K-Nearest Neighbors Regression</div>', unsafe_allow_html=True)

st.divider()

# 5. Sidebar Inputs
with st.sidebar:
    st.markdown("### 🎛️ Neural Parameters")
    st.markdown("Configure the input tensors:")
    
    number_courses = st.number_input(
        "📚 Number of Courses", 
        min_value=1, 
        max_value=20, 
        value=5, 
        step=1
    )

    time_study = st.number_input(
        "⏱️ Study Time (Hours)", 
        min_value=0.0, 
        max_value=168.0, 
        value=10.0, 
        step=0.5
    )

# 6. Data Preparation
input_data = pd.DataFrame({
    'number_courses': [number_courses],
    'time_study': [time_study]
})

with st.expander("👁️ View Extracted Input Tensor", expanded=False):
    st.dataframe(input_data, hide_index=True, use_container_width=True)

# 7. Prediction Logic
st.write("") 
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_btn = st.button("⚡ Initialize Prediction", type="primary", use_container_width=True)

if predict_btn:
    progress_text = "Processing data through model topology..."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(0.3)
    my_bar.empty() 
    
    try:
        prediction = model.predict(input_data)
        final_result = np.round(prediction.item(), 2)
        
        st.balloons()
        
        st.success("✅ Execution successful.")
        
        st.metric(
            label="🎯 Predicted Target Value", 
            value=str(final_result), 
            delta="KNN Model Output"
        )
        
    except Exception as e:
        st.error(f"System Error: {e}")
