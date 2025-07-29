import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model_cleaned.pkl")

model = load_model()

st.set_page_config(page_title="Employee Salary Estimator", layout="centered", page_icon="💼")

# Custom CSS to style the app
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #1668a7;
        transform: translateY(-2px);
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        font-size: 18px;
        margin: 1rem 0;
    }
    .feature-importance {
        margin-top: 2rem;
        padding: 1rem;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-text {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("💼 Employee Salary Estimator")
st.markdown("Use this tool to estimate an employee's expected monthly salary based on demographic and employment factors.")
st.markdown("---")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Personal Information
    st.subheader("Personal Information")
    age = st.slider("Age", 18, 90, 30, help="Employee's age in years")
    sex = st.selectbox("Gender", ["Female", "Male"], help="Employee's gender")
    race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], 
                       help="Employee's racial background")
    native_country = st.selectbox("Native Country", 
                                ["United-States", "Mexico", "Philippines", "Germany", "Canada", 
                                 "India", "China", "Cuba", "England", "Japan", "South"],
                                help="Employee's country of origin")

with col2:
    # Employment Information
    st.subheader("Employment Information")
    workclass = st.selectbox("Employment Sector", 
                           ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
                            "Local-gov", "State-gov", "Without-pay", "Never-worked"],
                           help="Employee's employment sector")
    education = st.selectbox("Education Level", 
                           ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", 
                            "Assoc-acdm", "Assoc-voc", "Doctorate", "7th-8th", "Prof-school", 
                            "5th-6th", "10th", "1st-4th", "Preschool", "12th"],
                           help="Employee's highest education level")
    education_num = st.slider("Education Years", 1, 16, 9, 
                            help="Number of years of education completed")
    hours_per_week = st.slider("Weekly Hours", 1, 99, 40, 
                              help="Average hours worked per week")

# Additional information in expandable section
with st.expander("Advanced Information"):
    marital_status = st.selectbox("Marital Status", 
                                ["Married-civ-spouse", "Divorced", "Never-married", 
                                 "Separated", "Widowed", "Married-spouse-absent", 
                                 "Married-AF-spouse"])
    occupation = st.selectbox("Occupation", 
                            ["Tech-support", "Craft-repair", "Other-service", "Sales", 
                             "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
                             "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
                             "Transport-moving", "Priv-house-serv", "Protective-serv", 
                             "Armed-Forces", "Unknown"])
    relationship = st.selectbox("Relationship Status", 
                              ["Wife", "Own-child", "Husband", "Not-in-family", 
                               "Other-relative", "Unmarried"])
    capital_gain = st.number_input("Capital Gain", 0, 
                                 help="Income from investment sources")
    capital_loss = st.number_input("Capital Loss", 0, 
                                  help="Losses from investment sources")
    fnlwgt = st.number_input("Final Weight", 0, 
                            help="Demographic weighting factor")

# Encode inputs
sex_encoded = 1 if sex == "Male" else 0

# Create a dictionary to map categories to encoded values
category_maps = {
    'workclass': ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
                 "Local-gov", "State-gov", "Without-pay", "Never-worked"],
    'education': ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", 
                 "Assoc-acdm", "Assoc-voc", "Doctorate", "7th-8th", "Prof-school", 
                 "5th-6th", "10th", "1st-4th", "Preschool", "12th"],
    'marital_status': ["Married-civ-spouse", "Divorced", "Never-married", 
                      "Separated", "Widowed", "Married-spouse-absent", 
                      "Married-AF-spouse"],
    'occupation': ["Tech-support", "Craft-repair", "Other-service", "Sales", 
                  "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
                  "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
                  "Transport-moving", "Priv-house-serv", "Protective-serv", 
                  "Armed-Forces", "Unknown"],
    'relationship': ["Wife", "Own-child", "Husband", "Not-in-family", 
                    "Other-relative", "Unmarried"],
    'race': ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    'native_country': ["United-States", "Mexico", "Philippines", "Germany", 
                      "Canada", "India", "China", "Cuba", "England", "Japan", "South"]
}

# Encode all categorical variables
encoded_inputs = {
    'age': age,
    'workclass': category_maps['workclass'].index(workclass),
    'education': category_maps['education'].index(education),
    'education_num': education_num,
    'marital_status': category_maps['marital_status'].index(marital_status),
    'occupation': category_maps['occupation'].index(occupation),
    'relationship': category_maps['relationship'].index(relationship),
    'race': category_maps['race'].index(race),
    'sex': sex_encoded,
    'capital_gain': capital_gain,
    'capital_loss': capital_loss,
    'hours_per_week': hours_per_week,
    'native_country': category_maps['native_country'].index(native_country),
    'fnlwgt': fnlwgt
}

# Convert to numpy array in the correct feature order
feature_order = [
    'age', 'workclass', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain',
    'capital_loss', 'hours_per_week', 'native_country', 'fnlwgt'
]

input_data = np.array([[encoded_inputs[feature] for feature in feature_order]])

# Prediction button
if st.button("Estimate Salary", key="predict_button"):
    with st.spinner("Calculating salary estimate..."):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        if prediction[0] == 1:
            salary = "₹75,000/month (High Income Group)"
            salary_value = 75000
            confidence = prediction_proba[0][1] * 100
        else:
            salary = "₹30,000/month (Lower Income Group)"
            salary_value = 30000
            confidence = prediction_proba[0][0] * 100
        
        st.success(f"""
        **Estimated Salary:** {salary}  
        **Prediction Confidence:** {confidence:.1f}%
        """)

        # Visualization section
        st.markdown("---")
        st.subheader("Salary Estimation Details")
        
        # Salary bar chart
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(["Predicted Salary"], [salary_value], color="#1f77b4")
        ax1.set_ylabel("Monthly Salary (INR)")
        ax1.set_ylim(0, 100000)
        ax1.set_title("Salary Estimate")
        st.pyplot(fig1)
        
        # Confidence meter
        st.write(f"**Confidence in Prediction:** {confidence:.1f}%")
        st.progress(int(confidence))
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.markdown("---")
            st.subheader("Key Factors Influencing This Prediction")
            
            # Get feature importances
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Create a DataFrame for display
            importance_df = pd.DataFrame({
                'Feature': [feature_order[i] for i in indices],
                'Importance': [importances[i] for i in indices]
            }).head(5)  # Show top 5 features
            
            # Plot feature importance
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Importance', y='Feature', data=importance_df, palette="Blues_d", ax=ax2)
            ax2.set_title("Top Influencing Factors")
            st.pyplot(fig2)
            
            # Show explanation of top features
            with st.expander("How these factors affect salary"):
                st.write("""
                - **Age:** Typically correlates with experience and higher salaries
                - **Education Level:** Higher education often leads to better-paying jobs
                - **Occupation:** Certain professions have higher earning potential
                - **Weekly Hours:** More hours worked can indicate higher compensation
                - **Employment Sector:** Government jobs often have different pay scales than private sector
                """)

st.markdown("</div>", unsafe_allow_html=True)