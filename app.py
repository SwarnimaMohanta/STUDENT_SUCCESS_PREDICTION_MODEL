import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import xgboost as xgb

# Set page config
st.set_page_config(page_title="Student Success Predictor", layout="wide")

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'student_success_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data')

# --- CACHED RESOURCES ---
@st.cache_resource
def load_resources():
    """
    Load the trained model, scaler, and the expected column structure.
    """
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # We need to reconstruct the training features to align columns
        mat_path = os.path.join(DATA_PATH, 'student-mat.csv')
        por_path = os.path.join(DATA_PATH, 'student-por.csv')
        
        df_mat = pd.read_csv(mat_path, sep=';')
        df_por = pd.read_csv(por_path, sep=';')
        df_mat['subject'] = 'Math'
        df_por['subject'] = 'Portuguese'
        df = pd.concat([df_mat, df_por], ignore_index=True)
        
        # Replicate preprocessing steps
        binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
        for col in binary_cols:
            df[col] = df[col].map({'yes': 1, 'no': 0})
            
        X = df.drop(['G3'], axis=1) # Original X
        X = pd.get_dummies(X, drop_first=True)
        
        expected_columns = X.columns.tolist()
        
        return model, scaler, expected_columns
    except FileNotFoundError:
        st.error("Model or Data files not found. Please run 'src/main.py' first.")
        return None, None, None

model, scaler, expected_columns = load_resources()

# --- UI HEADER ---
st.title("🎓 Student Success Prediction System")
st.markdown("""
This application uses **XGBoost** to predict whether a student will achieve a passing grade (G3 >= 10) 
based on their demographics, habits, and previous grades.
""")

if model is not None:
    
    # --- SIDEBAR INPUTS ---
    
    # 0. Personal Details (New Section)
    st.sidebar.header("Student Information")
    student_name = st.sidebar.text_input("Student Name", "John Doe")
    student_class = st.sidebar.text_input("Class / Section", "10-A")
    gender_ui = st.sidebar.selectbox("Gender", ["Female", "Male"])
    age = st.sidebar.number_input("Age", 15, 22, 16)
    
    # Map UI Gender to Dataset format ('F' or 'M')
    sex_val = 'F' if gender_ui == "Female" else 'M'

    # 1. Academic Info
    st.sidebar.subheader("Academic Performance")
    G1 = st.sidebar.slider("G1 Grade (1st INTERNAL)", 0, 20, 10)
    G2 = st.sidebar.slider("G2 Grade (2nd INTERNAL)", 0, 20, 10)
    failures = st.sidebar.number_input("Past Class Failures", 0, 4, 0)
    absences = st.sidebar.number_input("Number of Absences", 0, 93, 2)
    
    # 2. Study Habits
    st.sidebar.subheader("Study Habits")
    studytime = st.sidebar.select_slider("Weekly Study Time", options=[1, 2, 3, 4], format_func=lambda x: {1: "<2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: ">10 hrs"}[x])
    goout = st.sidebar.slider("Going Out (1=Low, 5=High)", 1, 5, 3)
    walc = st.sidebar.slider("Weekend Alcohol Consumption", 1, 5, 1)
    health = st.sidebar.slider("Health Status", 1, 5, 5)

    # 3. Demographics & Support
    st.sidebar.subheader("Demographics & Support")
    schoolsup = st.sidebar.selectbox("Extra Educational Support", ["no", "yes"])
    famsup = st.sidebar.selectbox("Family Educational Support", ["no", "yes"])
    higher = st.sidebar.selectbox("Wants Higher Education", ["yes", "no"])
    internet = st.sidebar.selectbox("Internet Access at Home", ["yes", "no"])
    reason = st.sidebar.selectbox("Reason for Choosing School", ["course", "home", "reputation", "other"])

    # --- PREDICTION LOGIC ---
    if st.sidebar.button("Predict Student Success"):
        
        # 1. Create Dictionary of Inputs
        input_data = {
            'school': 'GP', 
            'sex': sex_val,     # UPDATED: Uses user input
            'age': age,         # UPDATED: Uses user input
            'address': 'U', 
            'famsize': 'GT3', 
            'Pstatus': 'T', 
            'Medu': 4, 
            'Fedu': 4, 
            'Mjob': 'other', 
            'Fjob': 'other', 
            'reason': reason,
            'guardian': 'mother', 
            'traveltime': 1, 
            'studytime': studytime,
            'failures': failures,
            'schoolsup': schoolsup,
            'famsup': famsup,
            'paid': 'no',
            'activities': 'yes',
            'nursery': 'yes',
            'higher': higher,
            'internet': internet,
            'romantic': 'no',
            'famrel': 4,
            'freetime': 3,
            'goout': goout,
            'Dalc': 1,
            'Walc': walc,
            'health': health,
            'absences': absences,
            'G1': G1,
            'G2': G2,
            'subject': 'Math'
        }

        # 2. Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # 3. Preprocessing 
        binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
        for col in binary_cols:
            input_df[col] = input_df[col].map({'yes': 1, 'no': 0})

        # 4. One-Hot Encoding
        input_df = pd.get_dummies(input_df)

        # 5. ALIGN COLUMNS
        input_df_aligned = input_df.reindex(columns=expected_columns, fill_value=0)

        # 6. Scaling
        input_scaled = scaler.transform(input_df_aligned)

        # 7. Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # --- DISPLAY RESULTS ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Prediction for {student_name}")
            st.caption(f"Class: {student_class} | Age: {age}")
            
            if prediction == 1:
                st.success(f"PASS (Success) - Probability: {probability:.2%}")
                st.balloons()
            else:
                st.error(f"FAIL (At Risk) - Probability: {probability:.2%}")
                st.warning("Intervention Recommended")

        with col2:
            st.subheader("Key Drivers (Global)")
            fig, ax = plt.subplots(figsize=(6, 4))
            xgb.plot_importance(model, max_num_features=10, height=0.5, ax=ax, importance_type='weight', color='skyblue')
            st.pyplot(fig)
            
        # --- INTERPRETATION ---
        st.write("---")
        st.subheader("Why this result?")
        st.write(f"**{student_name}** has a **G2 grade of {G2}** and **{absences} absences**. The model combines these strong indicators with study time and family support to determine the likelihood of final success.")

else:
    st.info("Awaiting model files. Please run the training script first.")