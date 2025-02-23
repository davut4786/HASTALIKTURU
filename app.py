import streamlit as st
import pickle
import pandas as pd

# --------------------------
# Dosyaları Yükle
# --------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("xgb_modell.pkl", "rb") as f:
    model = pickle.load(f)

# --------------------------
# Uygulama Başlığı
# --------------------------
st.markdown("<h1 style='text-align: center;'>Hastalık Türü Tahmini</h1>", unsafe_allow_html=True)

# --------------------------
# Girdi Alanları İçin Ayarlar
# --------------------------
numeric_columns = [
    'cBaseEcfc', 'pCO2', 'pH', 'pHT', 'pO2', 'cCl', 'cK', 'cNa',
    'GRAN', 'LYM', 'LYM_A', 'MON', 'MON_A', 'Hb', 'HCT', 'MCH',
    'MCHC', 'MCV', 'MPV', 'PLT', 'RBC', 'RDW', 'WBC'
]

categorical_columns = {
    'hayvan_turu': ['kedi', 'kopek']
}

# Session state başlangıç değerleri
if 'numeric_inputs' not in st.session_state:
    st.session_state.numeric_inputs = {col: None for col in numeric_columns}

if 'categorical_inputs' not in st.session_state:
    st.session_state.categorical_inputs = {
        'hayvan_turu_kedi': 0,
        'hayvan_turu_kopek': 0
    }

# --------------------------
# Sayısal Girdiler (6 Sütunlu Düzen)
# --------------------------
st.markdown("**Laboratuvar Sonuçları**")
cols = st.columns(6)
col_index = 0

for i, col in enumerate(numeric_columns):
    current_col = cols[col_index]
    st.session_state.numeric_inputs[col] = current_col.number_input(
        label=col,
        value=None,
        format="%.2f",
        key=f"num_{col}"
    )
    col_index = (col_index + 1) % 6

# --------------------------
# Kategorik Girdiler
# --------------------------
st.markdown("**Hayvan Türü**")
animal_type = st.radio("Hayvan Türü Seçiniz", 
                       options=["Kedi", "Köpek"],
                       horizontal=True)

st.session_state.categorical_inputs = {
    'hayvan_turu_kedi': 1 if animal_type == "Kedi" else 0,
    'hayvan_turu_kopek': 1 if animal_type == "Köpek" else 0
}

# --------------------------
# Tahmin Butonu ve İşlemleri
# --------------------------
if st.button("Tahmin Et", type="primary"):
    # Veri birleştirme
    input_data = pd.DataFrame([{
        **st.session_state.numeric_inputs,
        **st.session_state.categorical_inputs
    }])
    
    # Eksik veri kontrolü
    missing_values = input_data.columns[input_data.isnull().any()].tolist()
    if missing_values:
        st.error(f"Eksik değerler bulundu: {', '.join(missing_values)}")
    else:
        try:
            # Ölçeklendirme
            scaled_data = scaler.transform(input_data[numeric_columns])
            input_data[numeric_columns] = scaled_data
            
            # Tahmin
            prediction = model.predict(input_data)[0]
            
            # Sonuçları eşleştirme
            disease_mapping = {
                0: "Enfeksiyöz",
                1: "Metablik",
                2: "Sağlıklı"
            }
            
            result = disease_mapping.get(prediction, "Bilinmeyen Durum")
            
            # Sonuç gösterimi
            st.success(f"Tahmini Sonuç: {result}")
            st.write("Detaylar:")
            st.dataframe(input_data.T.style.highlight_max(axis=0))
            
        except Exception as e:
            st.error(f"Hata oluştu: {str(e)}")
