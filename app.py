import streamlit as st
import pickle
import pandas as pd
import os

# --------------------------
# Dosyaları Yükle
# --------------------------
model_path = "model.pkl"  # Yeni model dosyasının adı

# Model dosyasının mevcut olup olmadığını kontrol et
if not os.path.exists(model_path):
    st.error("Model dosyası bulunamadı. Lütfen doğru dosya yolunu kontrol edin.")
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

# MinMaxScaler dosyasını yükleyin (Bu adımı da kontrol edelim)
scaler_path = "minmax_scaler.pkl"  # MinMaxScaler dosyasının adı
if not os.path.exists(scaler_path):
    st.error("Scaler dosyası bulunamadı. Lütfen doğru dosya yolunu kontrol edin.")
else:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

# --------------------------
# Uygulama Başlığı
# --------------------------
st.markdown("<h1 style='text-align: center;'>Hastalık Tahmin Uygulaması</h1>", unsafe_allow_html=True)

# --------------------------
# Girdi Alanları İçin Ayarlar
# --------------------------
columns = [
    "cBaseEcfc", "pCO2", "pH", "pHT", "pO2", "cCl", "cK", "cNa", "GRAN",
    "LYM", "LYM_A", "MON", "MON_A", "Hb", "HCT", "MCH", "MCHC", "MCV",
    "MPV", "PLT", "RBC", "RDW", "WBC"
]

# --------------------------
# Session state başlangıç değerleri
# --------------------------
if 'numeric_inputs' not in st.session_state:
    st.session_state.numeric_inputs = {col: None for col in columns}

if 'categorical_inputs' not in st.session_state:
    st.session_state.categorical_inputs = {
        "hayvan_turu_kedi": 0,
        "hayvan_turu_kopek": 0,
    }

# --------------------------
# Sayısal Girdiler (6 Sütunlu Düzen)
# --------------------------
st.markdown("**Hasta Bulguları ve Laboratuvar Sonuçları**")
col1, col2, col3, col4, col5, col6 = st.columns(6)
for i, col in enumerate(columns):
    key_val = f"{col}_input"
    if i % 6 == 0:
        st.session_state.numeric_inputs[col] = col1.number_input(col, value=None, format="%.2f", key=key_val)
    elif i % 6 == 1:
        st.session_state.numeric_inputs[col] = col2.number_input(col, value=None, format="%.2f", key=key_val)
    elif i % 6 == 2:
        st.session_state.numeric_inputs[col] = col3.number_input(col, value=None, format="%.2f", key=key_val)
    elif i % 6 == 3:
        st.session_state.numeric_inputs[col] = col4.number_input(col, value=None, format="%.2f", key=key_val)
    elif i % 6 == 4:
        st.session_state.numeric_inputs[col] = col5.number_input(col, value=None, format="%.2f", key=key_val)
    else:
        st.session_state.numeric_inputs[col] = col6.number_input(col, value=None, format="%.2f", key=key_val)

# --------------------------
# Hayvan Türü
# --------------------------
st.markdown("**Hayvan Türü**")
animal_type = st.radio("Hayvan Türü", options=["Kedi", "Köpek"],
                        index=0 if st.session_state.categorical_inputs["hayvan_turu_kedi"] == 1 else 1,
                        key="animal_type")
st.session_state.categorical_inputs["hayvan_turu_kedi"] = 1 if animal_type == "Kedi" else 0
st.session_state.categorical_inputs["hayvan_turu_kopek"] = 1 if animal_type == "Köpek" else 0

# --------------------------
# Tahmin Butonu ve İşlemleri
# --------------------------
if st.button("Tahmin Et"):
    input_data = pd.DataFrame([{**st.session_state.numeric_inputs, **st.session_state.categorical_inputs}])
    
    missing_columns = input_data.columns[input_data.isnull().any()].tolist()
    if missing_columns:
        st.warning("Lütfen şu sütunları doldurun: " + ", ".join(missing_columns))
    else:
        try:
            input_data[columns] = scaler.transform(input_data[columns])
            prediction = model.predict(input_data)[0]
            result_text = {0: "Enfeksiyoz", 1: "Metabolik", 2: "Sağlıklı"}[prediction]
            st.markdown(f"<h2 style='text-align: center;'>Tahmin Sonucu: {result_text}</h2>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")
