import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="AI Predictor Pro",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Fully Adaptive CSS using Native Streamlit Variables
st.markdown("""
    <style>
    /* 1. Animated Background using Streamlit's Native Theme Colors */
    /* This ensures the background matches the toggle menu perfectly */
    .stApp {
        background: linear-gradient(-45deg, var(--background-color), var(--secondary-background-color), var(--background-color));
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 2. Typewriter Text Animation */
    .typewriter-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .typewriter h1 {
        font-family: monospace;
        overflow: hidden;
        border-right: 0.15em solid var(--primary-color);
        white-space: nowrap;
        margin: 0 auto;
        letter-spacing: 0.1em;
        font-size: 2.5rem !important;
        /* Gradient using the theme's primary color and purple */
        background: -webkit-linear-gradient(var(--primary-color), #9333ea);
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
        50% { border-color: var(--primary-color); }
    }

    /* 3. Smooth Fade-In Subtext */
    .fade-in-text {
        text-align: center;
        color: var(--text-color);
        opacity: 0.7; /* Adapts cleanly to both light and dark backgrounds */
        font-size: 1.1rem;
        margin-bottom: 2rem;
        animation: fadeInUp 1.5s ease-out forwards;
        transform: translateY(20px);
    }

    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* 4. Adaptive Glassmorphism for Metric Cards */
    div[data-testid="metric-container"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 15px;
        padding: 25px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        /* Creates a glowing shadow based on the current Streamlit primary color */
        box-shadow: 0 0 15px var(--primary-color);
        border: 1px solid var(--primary-color);
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
    progress_text = "Processing data through model topology..."
    my_bar = st.progress(0, text=progress_text)
    
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(0.3)
    my_bar.empty() 
    
    try:
        # Generate prediction
        prediction = model.predict(input_data)
        final_result = np.round(prediction.item(), 2)
        
        # Trigger native animation
        st.balloons()
        
        # Display matching metric card
        st.success("✅ Execution successful.")
        st.metric(
            label="🎯 Predicted Target Value", 
            value=str(final_result), 
            delta="KNN Model Output"
        )
        
    except Exception as e:
        st.error(f"System Error: {e}")
