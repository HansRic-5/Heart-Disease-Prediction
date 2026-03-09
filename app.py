import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("artifacts/model.pkl")

def main():
    st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
    st.title('Heart Attack Risk Prediction')

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Umur', min_value=1, max_value=120, value=50)
        sex = st.selectbox('Jenis Kelamin', options=[0, 1], format_func=lambda x: 'Perempuan' if x == 0 else 'Laki-laki')
        cp = st.selectbox('Tipe Nyeri Dada (CP)', options=[0, 1, 2, 3], help="0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")
        fbs = st.selectbox('Gula Darah Puasa > 120 mg/dl', options=[0, 1])
        restecg = st.selectbox('Hasil EKG (Resting ECG)', options=[0, 1, 2])

    with col2:
        thalach = st.number_input('Detak Jantung Maksimal', value=150)
        exang = st.selectbox('Nyeri Dada Saat Olahraga?', options=[0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
        oldpeak = st.number_input('ST Depression (Oldpeak)', value=1.0, step=0.1)
        slope = st.selectbox('Slope ST Segment', options=[0, 1, 2])
        ca = st.selectbox('Jumlah Pembuluh Darah Utama (0-3)', options=[0, 1, 2, 3])
        thal = st.selectbox('Thalassemia', options=[1, 2, 3], help="1: Normal, 2: Fixed Defect, 3: Reversible Defect")

    st.divider()

    threshold = st.slider('Threshold Kewaspadaan', 0.0, 1.0, 0.5, 0.05)

    if st.button('Prediksi Risiko', use_container_width=True):
        data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': 120, 'chol': 200,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
            'ca': ca, 'thal': thal
        }
        
        input_df = pd.DataFrame([data])
        prob_sakit = model.predict_proba(input_df)[0][1]
        
        prediction = 1 if prob_sakit >= threshold else 0

        if prediction == 1:
            st.error(f"Resiko Tinggi (Probabilitas: {prob_sakit:.2%})")
        else:
            st.success(f"Resiko Rendah (Probabilitas: {prob_sakit:.2%})")

if __name__ == '__main__':
    main()