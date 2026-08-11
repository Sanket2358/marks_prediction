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

# 2. Fixed Light Mode CSS Injection
st.markdown("""
    <style>
    /* 1. Soft Light Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #f8fafc, #eff6ff, #e0e7ff, #f1f5f9);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 2. Typewriter Text - Vibrant & Readable */
    .typewriter-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .typewriter h1 {
        font-family: monospace;
        overflow: hidden;
        border-right: 0.15em solid #2563eb;
        white-space: nowrap;
        margin: 0 auto;
        letter-spacing: 0.1em;
        font-size: 2.5rem !important;
        background: -webkit-linear-gradient(#2563eb, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: 
            typing 2s steps(40, end),
            blink-caret 0.75s step-end infinite;
    }

    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }

    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: #2563eb; }
    }

    /* 3. Smooth Fade-In Subtext - Dark Slate Gray */
    .fade-in-text {
        text-align: center;
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        animation: fadeInUp 1.5s ease-out forwards;
        opacity: 0;
        transform: translateY(20px);
    }

    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* 4. Glassmorphism for Metric Cards - Soft Light Shadows */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 0, 0, 0.05);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.4s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(37, 99, 235, 0.3);
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
st.markdown("""
    <div class="typewriter-container">
        <div class="typewriter">
            <h1>AI Performance Engine</h1>
        </div>
    </div>
    <div class="fade-in-text">
        Powered by K-Nearest Neighbors Regression
    </div>
""", unsafe_allow_html=True)

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

# 7. Prediction Logic with UI sequence
st.write("") # Spacer
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_btn = st.button("⚡ Initialize Prediction", type="primary", use_container_width=True)

if predict_btn:
    # Add an advanced-looking progress bar
    progress_text = "Processing data through model topology..."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(0.3)
    my_bar.empty() # Clear the progress bar when done
    
    try:
        # Generate prediction
        prediction = model.predict(input_data)
        final_result = np.round(prediction.item(), 2)
        
        # Trigger native animation
        st.balloons()
        
        # Display Glassmorphism metric
        st.success("✅ Execution successful.")
        st.metric(
            label="🎯 Predicted Target Value", 
            value=str(final_result), 
            delta="KNN Model Output"
        )
        
    except Exception as e:
        st.error(f"System Error: {e}")
