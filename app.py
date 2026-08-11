import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Prediction Model",
    page_icon="📊",
    layout="centered"
)

# 2. Load the Model
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

# 3. Main Page Header
st.title("📊 Student Performance Predictor")
st.markdown("""
Welcome to the prediction app. Please adjust the parameters in the sidebar to match your current study habits, and click **Predict** to see the estimated outcome.
""")
st.divider()

# 4. Sidebar Inputs
st.sidebar.header("Input Features")
st.sidebar.markdown("Adjust the values below:")

# Extracting the features required by the model: number_courses and time_study
number_courses = st.sidebar.number_input(
    "Number of Courses", 
    min_value=1, 
    max_value=20, 
    value=5, 
    step=1,
    help="Total number of courses you are currently taking."
)

time_study = st.sidebar.number_input(
    "Time Study (Hours)", 
    min_value=0.0, 
    max_value=168.0, 
    value=10.0, 
    step=0.5,
    help="Total hours spent studying per week."
)

# 5. Data Preparation
input_data = pd.DataFrame({
    'number_courses': [number_courses],
    'time_study': [time_study]
})

st.subheader("Current Input Data")
st.dataframe(input_data, hide_index=True, use_container_width=True)

# 6. Prediction Logic
if st.button("🔮 Generate Prediction", type="primary"):
    with st.spinner("Calculating..."):
        try:
            # Generate prediction
            prediction = model.predict(input_data)
            
            # Safely extract and format the numerical result using numpy
            final_result = np.round(prediction.item(), 2)
            
            # Display result
            st.success("Prediction Complete!")
            st.metric(label="Predicted Target Value", value=str(final_result))
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
