import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import joblib
import matplotlib.pyplot as plt

# Load model dan scaler
model = load_model("model_ann.h5")
scaler = joblib.load("scaler.save")

# Load data
data = pd.read_csv("diabetes.csv")

st.title("📊 Perancangan Model Prediksi Diabetes Menggunakan Artificial Neural Network Berbasis Dataset Kesehatan")

# 1. Tampilkan Data Asli
st.subheader("1. Dataset Asli")
st.dataframe(data.head())

# 2. Struktur ANN
st.subheader("2. Visualisasi Model ANN")
st.image("ann_model.png", caption="Struktur Arsitektur ANN")

# Gambar ANN (jika ada)
if st.checkbox("Lihat gambar arsitektur ANN"):
    st.image("ann_model.png", caption="Struktur ANN")

# 3. Input User untuk Prediksi
st.subheader("3. Prediksi Risiko Baru")
input_data = {}

for col in data.columns[:-1]:
    input_data[col] = st.number_input(f"{col}", min_value=0.0)

if st.button("Prediksi"):
    input_array = np.array([list(input_data.values())])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    result = "✅ Berisiko Diabetes" if prediction[0][0] > 0.5 else "❌ Tidak Berisiko"
    st.success(f"Hasil Prediksi: {result} ({prediction[0][0]:.2f})")

# 4. Prediksi untuk Semua Data
st.subheader("4. Hasil Prediksi Dataset Asli")
X_data = scaler.transform(data.drop("Outcome", axis=1))
y_pred = model.predict(X_data)
data["Prediksi"] = (y_pred > 0.5).astype(int)
st.dataframe(data[["Outcome", "Prediksi"]])

# Akurasi Kasar
accuracy = (data["Outcome"] == data["Prediksi"]).mean()
st.write(f"🎯 Akurasi terhadap data asli: {accuracy * 100:.2f}%")
