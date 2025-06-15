import pandas as pd
import joblib
import streamlit as st

# Load trained SVM model
model = joblib.load("svm_model.pkl")

# Required input columns
expected_columns = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
    'Sex_M',
    'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
    'RestingECG_Normal', 'RestingECG_ST',
    'ExerciseAngina_Y',
    'ST_Slope_Flat', 'ST_Slope_Up'
]

# Streamlit UI setup
st.set_page_config(page_title="PulsePredict", layout="centered", page_icon=" ")
st.title(" PulsePredict - Arrhythmia Detection")
st.markdown("### Upload patient data or enter manually to check for arrhythmia risk.")

# File Upload Section
uploaded_file = st.file_uploader(" Upload CSV file with patient data", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Validate columns
        missing = [col for col in expected_columns if col not in df.columns]
        extra = [col for col in df.columns if col not in expected_columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if extra:
                st.warning(f" Extra columns detected and ignored: {extra}")
            df = df[expected_columns]  # Keep only expected

            # Predictions
            predictions = model.predict(df)
            df['ArrhythmiaPrediction'] = predictions
            df['Status'] = df['ArrhythmiaPrediction'].apply(
                lambda x: "Arrhythmia Detected" if x == 1 else "Heart is Safe")

            # Display results
            st.success(" Predictions complete!")
            st.dataframe(df[['Age', 'Status']])

            # Download
            st.download_button(
                " Download Results",
                data=df.to_csv(index=False),
                file_name="pulsepredict_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f" Error processing file: {e}")

else:
    st.subheader(" Or Enter Patient Details Manually")

    with st.form("manual_entry"):
        age = st.number_input("Age", 1, 120)
        resting_bp = st.number_input("Resting Blood Pressure", 0, 300)
        cholesterol = st.number_input("Cholesterol", 0, 600)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        max_hr = st.number_input("Maximum Heart Rate", 60, 250)
        oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, step=0.1)

        sex_m = st.selectbox("Sex", ["Male", "Female"]) == "Male"
        cp_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        ex_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"]) == "Yes"
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

        submitted = st.form_submit_button(" Predict")

        if submitted:
            row = {
                'Age': age,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': fasting_bs,
                'MaxHR': max_hr,
                'Oldpeak': oldpeak,
                'Sex_M': int(sex_m),
                'ChestPainType_ATA': int(cp_type == "ATA"),
                'ChestPainType_NAP': int(cp_type == "NAP"),
                'ChestPainType_TA': int(cp_type == "TA"),
                'RestingECG_Normal': int(rest_ecg == "Normal"),
                'RestingECG_ST': int(rest_ecg == "ST"),
                'ExerciseAngina_Y': int(ex_angina),
                'ST_Slope_Flat': int(st_slope == "Flat"),
                'ST_Slope_Up': int(st_slope == "Up")
            }

            input_df = pd.DataFrame([row])[expected_columns]
            prediction = model.predict(input_df)[0]
            proba = model.decision_function(input_df)[0]

            st.markdown("---")

            if prediction == 1:
                st.error(" **Arrhythmia Detected!** Immediate attention advised.")

                with st.expander(" Show Precautionary Measures"):
                    st.markdown("""
                    ###  Precautionary Measures You Should Take:
                    -  **Consult a Cardiologist:** Book an appointment immediately to discuss ECG and further diagnostics.
                    -  **Medication Compliance:** Follow prescribed medications strictly (e.g., beta-blockers, anticoagulants).
                    -  **Heart-Healthy Diet:** Low-sodium, high-fiber, avoid caffeine and processed foods.
                    -  **Avoid Triggers:** Minimize stress, smoking, alcohol, and overexertion.
                    -  **Practice Stress Reduction:** Yoga, meditation, and deep-breathing exercises help lower heart strain.
                    -  **Monitor Regularly:** Use wearable devices or follow-up with Holter monitoring.
                    -  **Controlled Physical Activity:** Engage in light to moderate walking or doctor-approved exercise.
                    -  **Control Underlying Conditions:** Manage diabetes, hypertension, and cholesterol levels.

                    **Emergency Warning Signs:**
                    > Chest pain, fainting, shortness of breath, or extreme fatigue? **Go to the hospital immediately.**
                    """)
            else:
                st.success(" **Heart is Safe** — No signs of arrhythmia.")

            # Disease probability
            confidence = abs(proba) / (abs(proba) + 1) * 100
            st.progress(min(int(confidence), 100))
            st.caption(f" Model confidence: `{confidence:.2f}%`")

            # Show Disease Percentage Button (separate)
            if st.button(" Show Disease Percentage"):
                disease_chance = confidence if prediction == 1 else 100 - confidence
                st.metric("Predicted Disease Chance", f"{disease_chance:.2f}%")

                if disease_chance >= 75:
                    st.warning(" High likelihood of arrhythmia. Consult a cardiologist.")
                elif 50 <= disease_chance < 75:
                    st.info(" Moderate risk. Consider regular monitoring.")
                else:
                    st.success(" Low risk. Keep maintaining a healthy lifestyle.")
