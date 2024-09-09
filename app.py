import streamlit as st
import pandas as pd
import joblib

# Set up the page configuration with a professional layout
st.set_page_config(
    page_title="Employee Performance Prediction",
    page_icon="🏢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a more polished and modern look
st.markdown("""
    <style>
    .stApp {
        background-color: #e9ecef;
    }
    body, .stTextInput, .stNumberInput, .stSelectbox {
        font-family: 'Helvetica', sans-serif;
        color: #343a40;
    }
    h1, h2, h3 {
        color: #0056b3;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .stAlert {
        background-color: #d1e7dd;
        border-left: 5px solid #198754;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stApp > div:first-child {
        display: flex;
        justify-content: center;
        align-items: center;
        padding-top: 2rem;
    }
    .footer {
        font-size: 12px;
        color: #adb5bd;
        text-align: center;
        padding: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained multi-output model
model = joblib.load('employee_multioutput_model.pkl')

# Streamlit app title with business-style heading
st.title("🔮 Employee Performance & Satisfaction Prediction")

# Header for input form
st.header("🔍 Enter Employee Details")

# Input fields with placeholders for guidance
with st.form(key='employee_form'):
    last_evaluation = st.number_input("🔍 Last Evaluation (0.0 - 1.0)", min_value=0.0, max_value=1.0, step=0.01,
                                      value=0.6)
    number_project = st.number_input("📈 Number of Projects (1 - 10)", min_value=1, max_value=10, step=1, value=4)
    average_montly_hours = st.number_input("⏲ Average Monthly Hours (80 - 320)", min_value=80, max_value=320, step=1,
                                           value=160)
    time_spend_company = st.number_input("🕒 Time Spent in Company (Years)", min_value=1, max_value=10, step=1, value=3)
    salary = st.selectbox("💼 Salary Level", options=['low', 'medium', 'high'], index=1)

    # Form submit button
    submit_button = st.form_submit_button(label='Predict Employee Outcomes')

# Convert salary level to numerical
salary_map = {'low': 0, 'medium': 1, 'high': 2}
salary_numeric = salary_map[salary]

# Predict button action
if submit_button:
    # Create a dataframe for prediction
    input_data = pd.DataFrame(
        [[last_evaluation, number_project, average_montly_hours, time_spend_company, salary_numeric]],
        columns=['last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'salary'])

    # Predict multiple outcomes (satisfaction, performance, promotion likelihood, attrition risk)
    predictions = model.predict(input_data)
    predicted_satisfaction, predicted_performance, predicted_promotion, predicted_attrition_risk = predictions[0]

    # Display the predicted outcomes with explanations
    st.success(f'🎯 Predicted Satisfaction: {predicted_satisfaction:.2f}')
    st.markdown("_Explanation: A higher value means the employee is more satisfied with their work environment._")

    st.success(f'🎯 Predicted Performance: {predicted_performance:.2f}')
    st.markdown("_Explanation: A higher score indicates stronger performance in recent evaluations._")

    st.success(f'🎯 Predicted Promotion Likelihood: {predicted_promotion:.2f}')
    st.markdown("_Explanation: A higher value suggests the employee has a better chance of being promoted._")

    st.success(f'🎯 Predicted Attrition Risk: {predicted_attrition_risk:.2f}')
    st.markdown("_Explanation: A higher score indicates the employee is more likely to leave the company._")

# Footer with branding
st.markdown("""
    <div class="footer">
        © 2024 Employee Analytics Co. | All Rights Reserved
    </div>
""", unsafe_allow_html=True)
