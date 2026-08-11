import streamlit as st
import pickle
import pandas as pd
import numpy as np
import time

# 1. Page Configuration
st.set_page_config(
    page_title="Prediction Model",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Animations and Styling
st.markdown("""
    <style>
    /* Animated Gradient Title */
    .animated-title {
        background: linear-gradient(-45deg, #FF4B4B, #FF8C42, #4CAF50, #2196F3);
        background-size: 400% 400%;
        animation: gradient 5s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Metric Card Hover Effect */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
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

# 4. Main Page Header
st.markdown('<div class="animated-title">Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Adjust the parameters in the sidebar and predict your estimated outcome! 🚀</p>", unsafe_allow_html=True)
st.divider()

# 5. Sidebar Inputs
with st.sidebar:
    st.header("🎛️ Input Features")
    st.markdown("Fine-tune your study habits:")
    
    number_courses = st.number_input(
        "📚 Number of Courses", 
        min_value=1, 
        max_value=20, 
        value=5, 
        step=1,
        help="Total number of courses you are currently taking."
    )

    time_study = st.number_input(
        "⏱️ Time Study (Hours)", 
        min_value=0.0, 
        max_value=168.0, 
        value=10.0, 
        step=0.5,
        help="Total hours spent studying per week."
    )

# 6. Data Preparation
input_data = pd.DataFrame({
    'number_courses': [number_courses],
    'time_study': [time_study]
})

# Use an expander to keep the UI clean
with st.expander("👀 View Current Input Data", expanded=True):
    st.dataframe(input_data, hide_index=True, use_container_width=True)

# 7. Prediction Logic
# Center the button using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔮 Generate Prediction", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Crunching the numbers... 🧮"):
        # Simulate a tiny delay for the animation effect
        time.sleep(0.75) 
        
        try:
            # Generate prediction
            prediction = model.predict(input_data)
            final_result = np.round(prediction.item(), 2)
            
            # Trigger Streamlit animations
            st.balloons()
            st.toast('Successfully calculated your score!', icon='✅')
            
            # Display result in a highlighted success box
            st.success("🎉 Prediction Complete!")
            
            # Display the animated metric card
            st.metric(label="Predicted Target Value", value=str(final_result), delta="Estimated Score")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
