# Fixed Streamlit App for Dropout Prediction Model with User-Friendly Input

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import warnings
warnings.filterwarnings('ignore')
import io

# Page configuration
st.set_page_config(
    page_title="Sistem Prediksi Dropout Risk",
    page_icon="⚠️",
    layout="wide"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .risk-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .risk-tinggi {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
    }
    
    .risk-sedang {
        background: linear-gradient(135deg, #ffa726, #ff9800);
        color: white;
    }
    
    .risk-rendah {
        background: linear-gradient(135deg, #66bb6a, #4caf50);
        color: white;
    }
    
    .metric-card {
        background: lightblue;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Define mapping dictionaries for user-friendly input
AGAMA_MAPPING = {
    'Islam': 1,
    'Kristen': 2,
    'Katolik': 3,
    'Hindu': 4,
    'Buddha': 5,
    'Lainnya': 6
}

PENDIDIKAN_MAPPING = {
    'Tidak sekolah': 1,
    'SD': 2,
    'SMP': 3,
    'SMA/SMK': 4,
    'D1/D2/D3': 5,
    'S1/S2/S3': 6
}

PEKERJAAN_MAPPING = {
    'Tidak bekerja': 1,
    'Petani': 2,
    'Buruh': 3,
    'Wiraswasta': 4,
    'PNS': 5,
    'TNI/Polri': 6,
    'Pensiunan': 7,
    'Nelayan': 8,
    'Pedagang': 9,
    'Ibu Rumah Tangga': 10
}

PENGHASILAN_MAPPING = {
    'Tidak berpenghasilan': 1,
    'Kurang dari Rp 1.000.000': 2,
    'Rp 1.000.000 - Rp 2.500.000': 3,
    'Rp 2.500.000 - Rp 5.000.000': 4,
    'Rp 5.000.000 - Rp 10.000.000': 5,
    'Lebih dari Rp 10.000.000': 6
}

JENIS_TINGGAL_MAPPING = {
    'Bersama orang tua': 1,
    'Bersama wali': 2,
    'Kost': 3,
    'Asrama': 4,
    'Panti asuhan': 5
}

TRANSPORTASI_MAPPING = {
    'Jalan kaki': 1,
    'Sepeda': 2,
    'Sepeda motor': 3,
    'Mobil pribadi': 4,
    'Angkutan umum': 5
}

TINGKAT_PENDIDIKAN_MAPPING = {
    'Kelas 7': 1,
    'Kelas 8': 2,
    'Kelas 9': 3
}

JENIS_PENDAFTARAN_MAPPING = {
    'Siswa baru': 1,
    'Pindahan': 2,
    'Naik kelas': 3
}

# Load model function
@st.cache_resource
def load_model():
    try:
        with open('dropout_prediction_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model file 'dropout_prediction_model.pkl' tidak ditemukan!")
        return None

# FIXED: Function to convert user input to model format
def convert_user_input_to_model_format(user_data):
    """
    Convert user-friendly input to model format with proper field mapping
    """
    converted_data = {}
    
    # Copy numeric fields directly
    numeric_fields = ['anak_keberapa', 'jumlah_saudara_kandung', 'jarak_rumah_ke_sekolah_km', 
                     'menit_tempuh_ke_sekolah', 'penerima_KPS', 'layak_PIP', 'penerima_KIP',
                     'a_pernah_paud', 'a_pernah_tk', 'kebutuhan_khusus_siswa', 
                     'kebutuhan_khusus_ayah', 'kebutuhan_khusus_ibu']
    
    for field in numeric_fields:
        if field in user_data:
            converted_data[field] = user_data[field]
    
    # Copy text fields that might be needed
    text_fields = ['nisn', 'nama', 'jenis_kelamin', 'bentuk_pendidikan', 'provinsi', 
                   'kabupaten', 'kecamatan', 'nama_ayah', 'nama_ibu']
    
    for field in text_fields:
        if field in user_data:
            converted_data[field] = user_data[field]
    
    # Convert categorical inputs to numeric IDs with proper field names
    if 'agama' in user_data:
        converted_data['agama_id'] = AGAMA_MAPPING.get(user_data['agama'], 1)
    
    if 'pendidikan_ayah' in user_data:
        converted_data['jenjang_pendidikan_ayah'] = PENDIDIKAN_MAPPING.get(user_data['pendidikan_ayah'], 1)
    
    if 'pendidikan_ibu' in user_data:
        converted_data['jenjang_pendidikan_ibu'] = PENDIDIKAN_MAPPING.get(user_data['pendidikan_ibu'], 1)
    
    if 'pekerjaan_ayah' in user_data:
        converted_data['pekerjaan_id_ayah'] = PEKERJAAN_MAPPING.get(user_data['pekerjaan_ayah'], 1)
    
    if 'pekerjaan_ibu' in user_data:
        converted_data['pekerjaan_id_ibu'] = PEKERJAAN_MAPPING.get(user_data['pekerjaan_ibu'], 1)
    
    if 'penghasilan_ayah' in user_data:
        converted_data['penghasilan_id_ayah'] = PENGHASILAN_MAPPING.get(user_data['penghasilan_ayah'], 1)
    
    if 'penghasilan_ibu' in user_data:
        converted_data['penghasilan_id_ibu'] = PENGHASILAN_MAPPING.get(user_data['penghasilan_ibu'], 1)
    
    if 'jenis_tinggal' in user_data:
        converted_data['jenis_tinggal_id'] = JENIS_TINGGAL_MAPPING.get(user_data['jenis_tinggal'], 1)
    
    if 'alat_transportasi' in user_data:
        converted_data['alat_transportasi_id'] = TRANSPORTASI_MAPPING.get(user_data['alat_transportasi'], 1)
    
    if 'tingkat_pendidikan' in user_data:
        converted_data['tingkat_pendidikan_id'] = TINGKAT_PENDIDIKAN_MAPPING.get(user_data['tingkat_pendidikan'], 1)
    
    if 'jenis_pendaftaran' in user_data:
        converted_data['jenis_pendaftaran_rombel'] = JENIS_PENDAFTARAN_MAPPING.get(user_data['jenis_pendaftaran'], 1)
    
    return converted_data

# FIXED: Improved prediction function with better error handling
def predict_dropout_risk(student_data, model_data):
    """
    Fungsi prediksi dropout risk dengan error handling yang lebih baik
    """
    try:
        # Extract model components
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoders = model_data.get('label_encoders', {})
        target_encoder = model_data.get('target_encoder', None)
        feature_names = model_data['feature_names']
        
        # Debug info
        st.write("🔍 **Debug Info:**")
        st.write(f"Feature names from model: {len(feature_names)}")
        st.write(f"Input data keys: {list(student_data.keys())}")
        
        # Convert student_data to DataFrame
        if isinstance(student_data, dict):
            df = pd.DataFrame([student_data])
        else:
            df = student_data.copy()
        
        # Create processed_data with all required features
        processed_data = pd.DataFrame()
        
        # Initialize all features with default values
        for feature in feature_names:
            if feature in df.columns:
                processed_data[feature] = df[feature]
            else:
                # Set appropriate default values based on feature type
                if 'id' in feature.lower() or feature.lower() in ['agama_id', 'jenjang_pendidikan_ayah', 'jenjang_pendidikan_ibu']:
                    processed_data[feature] = 1  # Default categorical ID
                elif feature in ['penerima_KPS', 'layak_PIP', 'penerima_KIP', 'a_pernah_paud', 'a_pernah_tk']:
                    processed_data[feature] = 0  # Default binary
                else:
                    processed_data[feature] = 0  # Default numeric
        
        st.write(f"Processed data shape: {processed_data.shape}")
        st.write(f"Processed data columns: {processed_data.columns.tolist()}")
        
        # Handle categorical encoding if label encoders exist
        for col, encoder in label_encoders.items():
            if col in processed_data.columns:
                try:
                    # Convert to string and handle unseen categories
                    values = processed_data[col].astype(str)
                    encoded_values = []
                    for val in values:
                        if val in encoder.classes_:
                            encoded_values.append(encoder.transform([val])[0])
                        else:
                            # Use the first class as default for unseen categories
                            encoded_values.append(encoder.transform([encoder.classes_[0]])[0])
                    processed_data[col] = encoded_values
                except Exception as e:
                    st.warning(f"Error encoding {col}: {str(e)}")
                    processed_data[col] = 0
        
        # Ensure proper data types
        processed_data = processed_data.astype(float)
        
        # Reorder columns to match training data
        processed_data = processed_data[feature_names]
        
        st.write(f"Final processed data shape: {processed_data.shape}")
        st.write(f"Sample values: {processed_data.iloc[0].head()}")
        
        # Scale features
        features_scaled = scaler.transform(processed_data)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        st.write(f"Raw prediction: {prediction}")
        st.write(f"Probabilities: {probabilities}")
        
        # Handle target encoding
        if target_encoder is not None:
            predicted_class = target_encoder.inverse_transform([prediction])[0]
            classes = target_encoder.classes_
        else:
            # Fallback if no target encoder
            predicted_class = 'DO' if prediction == 1 else 'LTM'
            classes = ['LTM', 'DO']  # Assume binary classification
        
        # Get dropout probability (assuming class 1 is DO)
        if len(probabilities) >= 2:
            if 'DO' in classes:
                do_idx = list(classes).index('DO')
                ltm_idx = list(classes).index('LTM')
                do_prob = probabilities[do_idx]
                ltm_prob = probabilities[ltm_idx]
            else:
                do_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                ltm_prob = probabilities[0] if len(probabilities) > 1 else 1 - probabilities[0]
        else:
            do_prob = probabilities[0]
            ltm_prob = 1 - probabilities[0]
        
        # Risk score
        risk_score = do_prob
        
        # Kategorisasi risiko
        if risk_score >= 0.7:
            category = 'Tinggi'
        elif risk_score >= 0.4:
            category = 'Sedang'
        else:
            category = 'Rendah'
        
        return {
            'predicted_class': predicted_class,
            'risk_score': risk_score,
            'risk_category': category,
            'do_probability': do_prob,
            'ltm_probability': ltm_prob
        }
        
    except Exception as e:
        st.error(f"Error dalam prediksi: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

# Main header
st.markdown("""
<div class="main-header">
    <h1>⚠️ Sistem Prediksi Dropout Risk Siswa</h1>
    <p>Prediksi risiko putus sekolah berdasarkan data siswa dan keluarga</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Menu Navigasi")
menu_option = st.sidebar.selectbox(
    "Pilih Menu:",
    ["Input Manual", "Upload Excel", "Debug Model", "Lihat Data Berisiko", "Statistik", "Input Intervensi", "Evaluasi Intervensi"]
)

# Load model
model_data = load_model()

if model_data is None:
    st.error("Model tidak dapat dimuat. Pastikan file 'dropout_prediction_model.pkl' tersedia.")
    st.stop()

# Show model info
with st.sidebar.expander("ℹ️ Info Model"):
    st.write(f"**Model Type:** {model_data.get('model_type', 'N/A')}")
    st.write(f"**AUC Score:** {model_data.get('auc_score', 'N/A'):.4f}")
    st.write(f"**Features:** {len(model_data.get('feature_names', []))}")

# NEW: Debug Model Menu
if menu_option == "Debug Model":
    st.header("🔧 Debug Model Information")
    
    # Display model structure
    st.subheader("Model Structure")
    for key, value in model_data.items():
        if key != 'model':  # Don't display the actual model object
            st.write(f"**{key}:** {type(value)}")
            if key == 'feature_names':
                st.write(f"Features ({len(value)}): {value}")
            elif key == 'label_encoders':
                st.write(f"Label encoders: {list(value.keys())}")
                for encoder_name, encoder in value.items():
                    st.write(f"  - {encoder_name}: {encoder.classes_}")
    
    # Test with sample data
    st.subheader("Test with Sample Data")
    if st.button("Test Sample Prediction"):
        sample_data = {
            'agama_id': 1,
            'jenjang_pendidikan_ayah': 4,
            'jenjang_pendidikan_ibu': 3,
            'pekerjaan_id_ayah': 2,
            'pekerjaan_id_ibu': 10,
            'penghasilan_id_ayah': 3,
            'penghasilan_id_ibu': 2,
            'jenis_tinggal_id': 1,
            'alat_transportasi_id': 3,
            'tingkat_pendidikan_id': 2,
            'jenis_pendaftaran_rombel': 1,
            'anak_keberapa': 2,
            'jumlah_saudara_kandung': 1,
            'jarak_rumah_ke_sekolah_km': 5.0,
            'menit_tempuh_ke_sekolah': 30,
            'penerima_KPS': 0,
            'layak_PIP': 0,
            'penerima_KIP': 0,
            'a_pernah_paud': 1,
            'a_pernah_tk': 1,
            'kebutuhan_khusus_siswa': 0,
            'kebutuhan_khusus_ayah': 0,
            'kebutuhan_khusus_ibu': 0
        }
        
        prediction = predict_dropout_risk(sample_data, model_data)
        if prediction:
            st.success("Sample prediction successful!")
            st.json(prediction)

# Menu: Input Manual (same as before, but with improved debugging)
elif menu_option == "Input Manual":
    st.header("📝 Input Data Siswa Manual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Pribadi")
        nisn = st.text_input("NISN")
        nama = st.text_input("Nama Lengkap")
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["L", "P"])
        agama = st.selectbox("Agama", list(AGAMA_MAPPING.keys()))
        anak_keberapa = st.number_input("Anak Keberapa", min_value=1, max_value=10, value=1)
        jumlah_saudara_kandung = st.number_input("Jumlah Saudara Kandung", min_value=0, max_value=10, value=0)
        
        st.subheader("Data Geografis")
        provinsi = st.text_input("Provinsi")
        kabupaten = st.text_input("Kabupaten")
        kecamatan = st.text_input("Kecamatan")
        jarak_rumah_ke_sekolah_km = st.number_input("Jarak Rumah ke Sekolah (km)", min_value=0.0, max_value=100.0, value=5.0)
        menit_tempuh_ke_sekolah = st.number_input("Menit Tempuh ke Sekolah", min_value=0, max_value=300, value=30)
        
        st.subheader("Data Tempat Tinggal")
        jenis_tinggal = st.selectbox("Jenis Tinggal", list(JENIS_TINGGAL_MAPPING.keys()))
        alat_transportasi = st.selectbox("Alat Transportasi", list(TRANSPORTASI_MAPPING.keys()))
    
    with col2:
        st.subheader("Data Bantuan Sosial")
        penerima_KPS = st.selectbox("Penerima KPS", ["Tidak", "Ya"])
        layak_PIP = st.selectbox("Layak PIP", ["Tidak", "Ya"])
        penerima_KIP = st.selectbox("Penerima KIP", ["Tidak", "Ya"])
        
        st.subheader("Data Orangtua")
        nama_ayah = st.text_input("Nama Ayah")
        pekerjaan_ayah = st.selectbox("Pekerjaan Ayah", list(PEKERJAAN_MAPPING.keys()))
        penghasilan_ayah = st.selectbox("Penghasilan Ayah", list(PENGHASILAN_MAPPING.keys()))
        pendidikan_ayah = st.selectbox("Pendidikan Ayah", list(PENDIDIKAN_MAPPING.keys()))
        
        nama_ibu = st.text_input("Nama Ibu")
        pekerjaan_ibu = st.selectbox("Pekerjaan Ibu", list(PEKERJAAN_MAPPING.keys()))
        penghasilan_ibu = st.selectbox("Penghasilan Ibu", list(PENGHASILAN_MAPPING.keys()))
        pendidikan_ibu = st.selectbox("Pendidikan Ibu", list(PENDIDIKAN_MAPPING.keys()))
        
        st.subheader("Data Pendidikan")
        bentuk_pendidikan = st.selectbox("Bentuk Pendidikan", ["SMA", "SMK", "MA"])
        tingkat_pendidikan = st.selectbox("Tingkat Pendidikan", list(TINGKAT_PENDIDIKAN_MAPPING.keys()))
        jenis_pendaftaran = st.selectbox("Jenis Pendaftaran", list(JENIS_PENDAFTARAN_MAPPING.keys()))
        a_pernah_paud = st.selectbox("Pernah PAUD", ["Tidak", "Ya"])
        a_pernah_tk = st.selectbox("Pernah TK", ["Tidak", "Ya"])
        
        st.subheader("Data Kebutuhan Khusus")
        kebutuhan_khusus_siswa = st.selectbox("Kebutuhan Khusus Siswa", ["Tidak", "Ya"])
        kebutuhan_khusus_ayah = st.selectbox("Kebutuhan Khusus Ayah", ["Tidak", "Ya"])
        kebutuhan_khusus_ibu = st.selectbox("Kebutuhan Khusus Ibu", ["Tidak", "Ya"])
    
    # Add debug mode toggle
    debug_mode = st.checkbox("🔍 Debug Mode", help="Show detailed processing information")
    
    if st.button("🔍 Prediksi Risiko Dropout", type="primary"):
        if nisn:
            # Prepare user-friendly data
            user_data = {
                'nisn': nisn,
                'nama': nama,
                'jenis_kelamin': jenis_kelamin,
                'agama': agama,
                'anak_keberapa': anak_keberapa,
                'jumlah_saudara_kandung': jumlah_saudara_kandung,
                'provinsi': provinsi,
                'kabupaten': kabupaten,
                'kecamatan': kecamatan,
                'jarak_rumah_ke_sekolah_km': jarak_rumah_ke_sekolah_km,
                'menit_tempuh_ke_sekolah': menit_tempuh_ke_sekolah,
                'jenis_tinggal': jenis_tinggal,
                'alat_transportasi': alat_transportasi,
                'penerima_KPS': 1 if penerima_KPS == "Ya" else 0,
                'layak_PIP': 1 if layak_PIP == "Ya" else 0,
                'penerima_KIP': 1 if penerima_KIP == "Ya" else 0,
                'nama_ayah': nama_ayah,
                'pekerjaan_ayah': pekerjaan_ayah,
                'penghasilan_ayah': penghasilan_ayah,
                'pendidikan_ayah': pendidikan_ayah,
                'nama_ibu': nama_ibu,
                'pekerjaan_ibu': pekerjaan_ibu,
                'penghasilan_ibu': penghasilan_ibu,
                'pendidikan_ibu': pendidikan_ibu,
                'bentuk_pendidikan': bentuk_pendidikan,
                'tingkat_pendidikan': tingkat_pendidikan,
                'jenis_pendaftaran': jenis_pendaftaran,
                'a_pernah_paud': 1 if a_pernah_paud == "Ya" else 0,
                'a_pernah_tk': 1 if a_pernah_tk == "Ya" else 0,
                'kebutuhan_khusus_siswa': 1 if kebutuhan_khusus_siswa == "Ya" else 0,
                'kebutuhan_khusus_ayah': 1 if kebutuhan_khusus_ayah == "Ya" else 0,
                'kebutuhan_khusus_ibu': 1 if kebutuhan_khusus_ibu == "Ya" else 0
            }
            
            if debug_mode:
                st.write("**Original user data:**")
                st.json(user_data)
            
            # Convert to model format
            model_data_input = convert_user_input_to_model_format(user_data)
            
            if debug_mode:
                st.write("**Converted model data:**")
                st.json(model_data_input)
            
            # Make prediction
            prediction = predict_dropout_risk(model_data_input, model_data)
            
            if prediction:
                st.success("✅ Prediksi berhasil!")
                
                # Display student info
                st.subheader(f"Hasil Prediksi untuk {nama} (NISN: {nisn})")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                                
                    <div class="risk-card risk-{prediction['risk_category'].lower()}">
                        <h3>Kategori Risiko</h3>
                        <h2>{prediction['risk_category']}</h2>
                        <p>Skor Risiko: {prediction['risk_score']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Prediksi Status</h3>
                        <h2>{prediction['predicted_class']}</h2>
                        <p>{'Dropout' if prediction['predicted_class'] == 'DO' else 'Lulus Tepat Waktu'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Probabilitas Detail</h3>
                        <p>Dropout: {prediction['do_probability']:.2%}</p>
                        <p>Lulus Tepat Waktu: {prediction['ltm_probability']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction['risk_score'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Skor Risiko Dropout (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation
                st.subheader("📋 Interpretasi Hasil")
                if prediction['risk_category'] == 'Tinggi':
                    st.warning("⚠️ **Risiko Tinggi**: Siswa memiliki risiko tinggi untuk dropout. Perlu intervensi segera.")
                elif prediction['risk_category'] == 'Sedang':
                    st.info("ℹ️ **Risiko Sedang**: Siswa memiliki risiko sedang. Perlu monitoring dan dukungan.")
                else:
                    st.success("✅ **Risiko Rendah**: Siswa memiliki risiko rendah untuk dropout.")
                
        else:
            st.error("Mohon masukkan NISN siswa")

if menu_option == "Upload Excel":
    st.header("📤 Upload Data Excel")
    
    # Template download
    st.subheader("📥 Download Template")
    if st.button("Download Template Excel"):
        # Buat template Excel dengan format yang sesuai dengan input manual
        template_data = {
            'nisn': ['300023', '300024', '300025', '300026', '300027'],
            'nama': ['Ahmad Fauzi', 'Siti Nurhaliza', 'Budi Santoso', 'Dewi Sartika', 'Rizky Pratama'],
            'jenis_kelamin': ['L', 'P', 'L', 'P', 'L'],
            'agama': ['Islam', 'Islam', 'Kristen', 'Hindu', 'Islam'],
            'anak_keberapa': [1, 1, 1, 1, 1],
            'jumlah_saudara_kandung': [1, 1, 1, 1, 1],
            'provinsi': ['Banten', 'Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur'],
            'kabupaten': ['Serang', 'Jakarta Pusat', 'Bandung', 'Semarang', 'Surabaya'],
            'kecamatan': ['Cinangka', 'Kemayoran', 'Antapani', 'Banyumanik', 'Tegalsari'],
            'jarak_rumah_ke_sekolah_km': [1, 3.2, 1, 2.1, 4.8],
            'menit_tempuh_ke_sekolah': [3, 20, 2, 15, 25],
            'jenis_tinggal': ['Bersama orang tua', 'Bersama orang tua', 'Kost', 'Bersama orang tua', 'Bersama orang tua'],
            'alat_transportasi': ['Sepeda motor', 'Jalan kaki', 'Angkutan umum', 'Sepeda', 'Sepeda motor'],
            'penerima_KPS': ['Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak'],
            'layak_PIP': ['Ya', 'Ya', 'Tidak', 'Ya', 'Tidak'],
            'penerima_KIP': ['Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak'],
            'nama_ayah': ['Suryadi', 'Bambang', 'Hendra', 'Agus', 'Imam'],
            'pekerjaan_ayah': ['Petani', 'Wiraswasta', 'Nelayan', 'Guru', 'Buruh'],
            'penghasilan_ayah': ['Kurang dari Rp. 500,000', 'Rp. 500,000 - Rp. 999,999', 'Rp. 1,000,000 - Rp. 1,999,999', 'Rp. 2,000,000 - Rp. 4,999,999', 'Rp. 500,000 - Rp. 999,999'],
            'pendidikan_ayah': ['SD / sederajat', 'SMP / sederajat', 'SMA / sederajat', 'S1', 'SMP / sederajat'],
            'nama_ibu': ['Siti Aisyah', 'Marni', 'Sri Wahyuni', 'Dewi', 'Ratna'],
            'pekerjaan_ibu': ['Ibu rumah tangga', 'Buruh', 'Petani', 'Guru', 'Pedagang'],
            'penghasilan_ibu': ['Rp. 1,000,000 - Rp. 1,999,999', 'Kurang dari Rp. 500,000', 'Kurang dari Rp. 500,000', 'Rp. 1,000,000 - Rp. 1,999,999', 'Rp. 500,000 - Rp. 999,999'],
            'pendidikan_ibu': ['SMA', 'SMP / sederajat', 'SMA / sederajat', 'S1', 'SMP / sederajat'],
            'bentuk_pendidikan': ['SMA', 'SMA', 'SMK', 'SMA', 'SMK'],
            'tingkat_pendidikan': ['Kelas 10', 'Kelas 11', 'Kelas 12', 'Kelas 10', 'Kelas 11'],
            'jenis_pendaftaran': ['Siswa baru', 'Siswa baru', 'Siswa baru', 'Siswa baru', 'Siswa baru'],
            'a_pernah_paud': ['Ya', 'Tidak', 'Ya', 'Ya', 'Tidak'],
            'a_pernah_tk': ['Ya', 'Ya', 'Tidak', 'Ya', 'Ya'],
            'kebutuhan_khusus_siswa': ['Tidak', 'Tidak', 'Tidak', 'Tidak', 'Tidak'],
            'kebutuhan_khusus_ayah': ['Tidak', 'Tidak', 'Tidak', 'Tidak', 'Tidak'],
            'kebutuhan_khusus_ibu': ['Tidak', 'Tidak', 'Tidak', 'Tidak', 'Tidak']
        }
        
        df_template = pd.DataFrame(template_data)
        
        # Convert to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_template.to_excel(writer, sheet_name='Template', index=False)
        
        st.download_button(
            label="📥 Download Template Excel",
            data=output.getvalue(),
            file_name="template_prediksi_dropout.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # File upload
    st.subheader("📤 Upload File Excel")
    uploaded_file = st.file_uploader("Pilih file Excel", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ File berhasil diupload! Jumlah data: {len(df)} siswa")
            
            # Tampilkan preview data
            st.subheader("👁️ Preview Data")
            st.dataframe(df.head())
            
            # Validasi kolom yang diperlukan
            required_columns = [
                'nisn', 'nama', 'jenis_kelamin', 'agama', 'anak_keberapa', 'jumlah_saudara_kandung',
                'provinsi', 'kabupaten', 'kecamatan', 'jarak_rumah_ke_sekolah_km', 'menit_tempuh_ke_sekolah',
                'jenis_tinggal', 'alat_transportasi', 'penerima_KPS', 'layak_PIP', 'penerima_KIP',
                'nama_ayah', 'pekerjaan_ayah', 'penghasilan_ayah', 'pendidikan_ayah',
                'nama_ibu', 'pekerjaan_ibu', 'penghasilan_ibu', 'pendidikan_ibu',
                'bentuk_pendidikan', 'tingkat_pendidikan', 'jenis_pendaftaran',
                'a_pernah_paud', 'a_pernah_tk', 'kebutuhan_khusus_siswa',
                'kebutuhan_khusus_ayah', 'kebutuhan_khusus_ibu'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Kolom yang hilang: {', '.join(missing_columns)}")
                st.info("📝 Silakan download template untuk melihat format yang benar")
            else:
                # Add debug mode for batch processing
                debug_mode = st.checkbox("🔍 Debug Mode (Batch)", help="Show detailed processing information for batch")
                
                # Proses prediksi batch
                if st.button("🔍 Proses Prediksi Batch", type="primary") and model_data:
                    progress_bar = st.progress(0)
                    results = []
                    
                    for idx, row in df.iterrows():
                        # Update progress
                        progress_bar.progress((idx + 1) / len(df))
                        
                        try:
                            # Konversi data seperti pada input manual
                            user_data = {
                                'nisn': str(row['nisn']),
                                'nama': str(row['nama']),
                                'jenis_kelamin': row['jenis_kelamin'],
                                'agama': row['agama'],
                                'anak_keberapa': int(row['anak_keberapa']),
                                'jumlah_saudara_kandung': int(row['jumlah_saudara_kandung']),
                                'provinsi': str(row['provinsi']),
                                'kabupaten': str(row['kabupaten']),
                                'kecamatan': str(row['kecamatan']),
                                'jarak_rumah_ke_sekolah_km': float(row['jarak_rumah_ke_sekolah_km']),
                                'menit_tempuh_ke_sekolah': int(row['menit_tempuh_ke_sekolah']),
                                'jenis_tinggal': row['jenis_tinggal'],
                                'alat_transportasi': row['alat_transportasi'],
                                'penerima_KPS': 1 if row['penerima_KPS'] == "Ya" else 0,
                                'layak_PIP': 1 if row['layak_PIP'] == "Ya" else 0,
                                'penerima_KIP': 1 if row['penerima_KIP'] == "Ya" else 0,
                                'nama_ayah': str(row['nama_ayah']),
                                'pekerjaan_ayah': row['pekerjaan_ayah'],
                                'penghasilan_ayah': row['penghasilan_ayah'],
                                'pendidikan_ayah': row['pendidikan_ayah'],
                                'nama_ibu': str(row['nama_ibu']),
                                'pekerjaan_ibu': row['pekerjaan_ibu'],
                                'penghasilan_ibu': row['penghasilan_ibu'],
                                'pendidikan_ibu': row['pendidikan_ibu'],
                                'bentuk_pendidikan': row['bentuk_pendidikan'],
                                'tingkat_pendidikan': row['tingkat_pendidikan'],
                                'jenis_pendaftaran': row['jenis_pendaftaran'],
                                'a_pernah_paud': 1 if row['a_pernah_paud'] == "Ya" else 0,
                                'a_pernah_tk': 1 if row['a_pernah_tk'] == "Ya" else 0,
                                'kebutuhan_khusus_siswa': 1 if row['kebutuhan_khusus_siswa'] == "Ya" else 0,
                                'kebutuhan_khusus_ayah': 1 if row['kebutuhan_khusus_ayah'] == "Ya" else 0,
                                'kebutuhan_khusus_ibu': 1 if row['kebutuhan_khusus_ibu'] == "Ya" else 0
                            }
                            
                            if debug_mode:
                                st.write(f"**Processing row {idx + 1}:**")
                                st.write("Original data:", user_data)
                            
                            # Convert to model format
                            model_data_input = convert_user_input_to_model_format(user_data)
                            
                            if debug_mode:
                                st.write("Converted model data:", model_data_input)
                            
                            # Prediksi
                            prediction = predict_dropout_risk(model_data_input, model_data)
                            
                            if prediction:
                                results.append({
                                    'nisn': user_data['nisn'],
                                    'nama': user_data['nama'],
                                    'jenis_kelamin': user_data['jenis_kelamin'],
                                    'kelas': user_data['tingkat_pendidikan'],
                                    'skor_risiko': prediction['risk_score'],
                                    'kategori_risiko': prediction['risk_category'],
                                    'prediksi_status': prediction['predicted_class'],
                                    'probabilitas_dropout': prediction['do_probability'],
                                    'probabilitas_lulus': prediction['ltm_probability']
                                })
                                
                                if debug_mode:
                                    st.write("Prediction result:", prediction)
                            else:
                                st.warning(f"⚠️ Gagal memprediksi baris {idx+1} - {user_data['nama']}")
                                
                        except Exception as e:
                            st.error(f"❌ Error pada baris {idx+1}: {str(e)}")
                            if debug_mode:
                                st.write("Error details:", str(e))
                            continue
                    
                    # Simpan hasil
                    if results:
                        df_results = pd.DataFrame(results)
                        st.session_state.batch_results = df_results
                        
                        st.success(f"✅ Prediksi batch selesai! Berhasil memproses {len(results)} dari {len(df)} siswa")
                        
                        # Tampilkan statistik hasil
                        st.subheader("📊 Ringkasan Hasil Prediksi")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            tinggi = len(df_results[df_results['kategori_risiko'] == 'Tinggi'])
                            st.metric("Risiko Tinggi", tinggi)
                        
                        with col2:
                            sedang = len(df_results[df_results['kategori_risiko'] == 'Sedang'])
                            st.metric("Risiko Sedang", sedang)
                        
                        with col3:
                            rendah = len(df_results[df_results['kategori_risiko'] == 'Rendah'])
                            st.metric("Risiko Rendah", rendah)
                        
                        # Tampilkan hasil detail
                        st.subheader("📋 Hasil Prediksi Detail")
                        st.dataframe(df_results)
                        
                        # Download hasil
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_results.to_excel(writer, sheet_name='Hasil Prediksi', index=False)
                        
                        st.download_button(
                            label="📥 Download Hasil Prediksi",
                            data=output.getvalue(),
                            file_name=f"hasil_prediksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.error("❌ Tidak ada data yang berhasil diproses")
                        
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")
            st.info("💡 Pastikan file Excel sesuai dengan template yang disediakan")

elif menu_option == "Lihat Data Berisiko":
    st.header("Daftar Siswa Berisiko")
    
    # Simulasi data siswa berisiko
    if 'batch_results' in st.session_state:
        df_results = st.session_state.batch_results
        
        # Filter berdasarkan kategori risiko
        risk_filter = st.selectbox("Filter berdasarkan risiko:", ["Semua", "Tinggi", "Sedang", "Rendah"])
        
        if risk_filter != "Semua":
            df_filtered = df_results[df_results['kategori_risiko'] == risk_filter]
        else:
            df_filtered = df_results
        
        # Tampilkan dalam format seperti gambar
        st.subheader(f"📋 Daftar Siswa Berisiko ({risk_filter})")
        
        # Buat tabel dengan styling sesuai gambar
        for idx, row in df_filtered.iterrows():
            risk_color = "🔴" if row['kategori_risiko'] == "Tinggi" else "🟡" if row['kategori_risiko'] == "Sedang" else "🟢"
            
            col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 1, 1])
            
            with col1:
                st.write(f"**{idx}**")
            with col2:
                st.write(f"**{row['nisn']}**")
            with col3:
                st.write(f"**{row['nama_siswa']}**")
            with col4:
                st.write(f"**{row['kelas']}**")
            with col5:
                st.write(f"**{risk_color} {row['skor_risiko']:.0f}**")
            
            st.divider()
    else:
        st.info("📝 Belum ada data prediksi. Silakan upload file Excel terlebih dahulu.")

elif menu_option == "Statistik":
    st.header("📈 Statistik Dropout Risk")
    
    if 'batch_results' in st.session_state:
        df_results = st.session_state.batch_results
        
        # Statistik dasar
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_siswa = len(df_results)
            st.metric("Total Siswa", total_siswa)
            
        with col2:
            high_risk = len(df_results[df_results['kategori_risiko'] == 'Tinggi'])
            st.metric("Risiko Tinggi", high_risk, f"{high_risk/total_siswa*100:.1f}%")
            
        with col3:
            medium_risk = len(df_results[df_results['kategori_risiko'] == 'Sedang'])
            st.metric("Risiko Sedang", medium_risk, f"{medium_risk/total_siswa*100:.1f}%")
            
        with col4:
            low_risk = len(df_results[df_results['kategori_risiko'] == 'Rendah'])
            st.metric("Risiko Rendah", low_risk, f"{low_risk/total_siswa*100:.1f}%")
        
        # Chart distribusi risiko
        st.subheader("📊 Distribusi Kategori Risiko")
        risk_counts = df_results['kategori_risiko'].value_counts()
        st.bar_chart(risk_counts)
        
        # Chart skor risiko
        fig = px.histogram(df_results, x='skor_risiko', nbins=20, title='Distribusi Skor Risiko')
        st.plotly_chart(fig)
        
    else:
        st.info("📝 Belum ada data untuk ditampilkan. Silakan upload file Excel terlebih dahulu.")

elif menu_option == "Input Intervensi":
    st.header("Input Intervensi Sekolah")
    def show_input_intervensi():
 
        st.markdown("""
        Input intervensi yang dilakukan
        """)

        df = pd.read_csv("/Users/yuniaameliachairunisa/Documents/Project-EWS-Pusdatin/data/ews_dummy_data/data_siswa.csv")
        id_siswa = st.selectbox("Pilih ID Siswa", df['id_siswa'])

        jenis = st.selectbox("Jenis Intervensi", ["Kunjungan rumah", "Konseling", "Remedial", "Ajukan KIP/PIP"])
        hasil = st.selectbox("Hasil", ["Masih Berisiko", "Lanjut Sekolah", "Putus Sekolah"])
        tanggal = st.date_input("Tanggal Intervensi", value=date.today())

        if st.button("Simpan"):
            baru = pd.DataFrame([{
                "id_siswa": id_siswa,
                "jenis_intervensi": jenis,
                "tanggal_intervensi": tanggal,
                "hasil": hasil
            }])
            intervensi = pd.read_csv("/Users/yuniaameliachairunisa/Documents/Project-EWS-Pusdatin/data/ews_dummy_data/data_intervensi.csv")
            intervensi = pd.concat([intervensi, baru])
            intervensi.to_csv("data/data_intervensi.csv", index=False)
            st.success("Intervensi berhasil disimpan.")
    show_input_intervensi()

elif menu_option == "Evaluasi Intervensi":
    def show_evaluasi_intervensi():
        df1 = pd.read_csv("/Users/yuniaameliachairunisa/Documents/Project-EWS-Pusdatin/data/ews_dummy_data/data_intervensi.csv")
        hasil = df1.groupby("hasil").size().reset_index(name="jumlah")
        fig = px.pie(hasil, names="hasil", values="jumlah", title="Distribusi Hasil Intervensi")
        st.plotly_chart(fig)
    show_evaluasi_intervensi()
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 1rem;">
    <p>🏫 Sistem Prediksi Dropout Risk Siswa - Dikembangkan untuk pendidikan yang lebih baik</p>
    <p>📧 Untuk pertanyaan teknis, hubungi tim IT sekolah</p>
</div>
""", unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("💡 **Sistem Prediksi Dropout Risk** - Membantu identifikasi dini siswa berisiko putus sekolah")