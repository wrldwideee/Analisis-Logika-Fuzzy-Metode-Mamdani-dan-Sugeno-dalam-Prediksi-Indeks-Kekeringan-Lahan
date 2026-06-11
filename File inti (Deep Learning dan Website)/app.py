import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from fuzzy_logic import FuzzyKekeringan

# Konfigurasi Halaman (Harus di awal)
st.set_page_config(page_title="Prediksi Kekeringan", page_icon="🏜️", layout="wide")

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .highlight {
        background: linear-gradient(120deg, #84fab0 0%, #8fd3f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .card {
        background-color: #1E2127;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('climate_data.csv')
    df = df.ffill().bfill()
    cols = ['Tavg', 'RH_avg', 'RR', 'ss', 'ff_avg']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.ffill().bfill()
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Gagal memuat dataset 'climate_data.csv'. Pastikan file berada di direktori yang sama.")
    st.stop()

fuzzy_sys = FuzzyKekeringan()

def plot_fuzzy_results(f_vals, agg, hasil_mamdani, fuzzy_sys_obj):
    st.markdown("---")
    st.markdown("<h3 class='highlight'>Visualisasi Keputusan Fuzzy</h3>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("**1. Heatmap Fuzzifikasi (Derajat Keanggotaan)**")
        data = []
        labels = ['Suhu', 'Kelembapan', 'Hujan', 'Cahaya', 'Angin']
        terms = ['Rendah', 'Sedang', 'Tinggi']
        for lbl in labels:
            data.append([f_vals[lbl]['Rendah'], f_vals[lbl]['Sedang'], f_vals[lbl]['Tinggi']])
        
        fig, ax = plt.subplots(figsize=(6, 4))
        # Dark theme for heatmap to match the website
        fig.patch.set_facecolor('#1E2127')
        ax.set_facecolor('#1E2127')
        sns.heatmap(data, annot=True, cmap="magma", xticklabels=terms, yticklabels=labels, ax=ax, vmin=0, vmax=1)
        ax.tick_params(colors='white')
        st.pyplot(fig)
        
    with colB:
        st.markdown("**2. Area Defuzzifikasi Mamdani**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#1E2127')
        ax2.set_facecolor('#1E2127')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('white')
        
        y_range = fuzzy_sys_obj.y_range
        out_params = {'Rendah': [-50, 0, 50], 'Sedang': [10, 50, 90], 'Tinggi': [50, 100, 150]}
        
        ax2.plot(y_range, fuzzy_sys_obj.trimf(y_range, out_params['Rendah']), color='#8fd3f4', linestyle='--', alpha=0.5, label='Rendah')
        ax2.plot(y_range, fuzzy_sys_obj.trimf(y_range, out_params['Sedang']), color='#84fab0', linestyle='--', alpha=0.5, label='Sedang')
        ax2.plot(y_range, fuzzy_sys_obj.trimf(y_range, out_params['Tinggi']), color='#ff6b6b', linestyle='--', alpha=0.5, label='Tinggi')
        
        shaded_y = [max(
            min(agg['Rendah'], fuzzy_sys_obj.trimf(y, out_params['Rendah'])),
            min(agg['Sedang'], fuzzy_sys_obj.trimf(y, out_params['Sedang'])),
            min(agg['Tinggi'], fuzzy_sys_obj.trimf(y, out_params['Tinggi']))
        ) for y in y_range]
            
        ax2.fill_between(y_range, 0, shaded_y, color='#a777e3', alpha=0.8, label='Area Agregasi')
        ax2.axvline(x=hasil_mamdani, color='white', linestyle='-', linewidth=2, label=f'Centroid: {hasil_mamdani:.1f}')
        
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 1.1)
        legend = ax2.legend(loc='upper right', fontsize='small')
        for text in legend.get_texts():
            text.set_color('black')
        st.pyplot(fig2)

# Navigasi Sidebar
st.sidebar.title("🏜️ Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman", [
    "Beranda & Dataset", 
    "Sistem Fuzzy (Manual)", 
    "Integrasi Deep Learning (Forecasting)"
])

st.sidebar.markdown("---")
st.sidebar.info(
    "Aplikasi ini dikembangkan untuk memenuhi syarat tambahan "
    "Membangun Web Streamlit dan Integrasi Deep Learning (Neural Network)."
)

if menu == "Beranda & Dataset":
    st.markdown("<h1 class='highlight'>Analisis Tingkat Kekeringan Harian BMKG</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h3>Selamat Datang di Sistem Prediksi Kekeringan</h3>
        <p>Aplikasi ini memadukan <b>Logika Fuzzy</b> murni (Mamdani & Sugeno) dengan <b>Deep Learning</b> (Multi-Layer Perceptron Neural Network) untuk memprediksi tingkat kekeringan berdasarkan data historis stasiun iklim harian BMKG.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Cuplikan Data Iklim Historis (100 Terakhir)")
    st.dataframe(df.tail(100), use_container_width=True)
    
    st.subheader("Tren Suhu & Kelembapan")
    chart_data = df[['Tavg', 'RH_avg']].tail(100)
    st.line_chart(chart_data)

elif menu == "Sistem Fuzzy (Manual)":
    st.markdown("<h1 class='highlight'>Simulasi Fuzzy Kekeringan</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>Silakan sesuaikan parameter iklim pada slider di bawah ini untuk melihat bagaimana Logika Fuzzy secara langsung (secara manual) memproses nilai crisp menjadi Indeks Kekeringan menggunakan aturan dasar (15 rules).</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        suhu = st.slider("Suhu Rata-rata (Tavg)", float(df['Tavg'].min()), float(df['Tavg'].max()), float(df['Tavg'].mean()))
        kelembapan = st.slider("Kelembapan Rata-rata (RH_avg)", float(df['RH_avg'].min()), float(df['RH_avg'].max()), float(df['RH_avg'].mean()))
    with col2:
        hujan = st.slider("Curah Hujan (RR)", float(df['RR'].min()), float(df['RR'].max()), float(df['RR'].mean()))
        cahaya = st.slider("Lama Penyinaran (ss)", float(df['ss'].min()), float(df['ss'].max()), float(df['ss'].mean()))
    with col3:
        angin = st.slider("Kecepatan Angin (ff_avg)", float(df['ff_avg'].min()), float(df['ff_avg'].max()), float(df['ff_avg'].mean()))
        
    st.markdown("---")
    if st.button("Hitung Tingkat Kekeringan (Inferensi Fuzzy)"):
        # Normalisasi menggunakan min max dari dataset
        def norm(val, col):
            return (val - df[col].min()) / (df[col].max() - df[col].min())
            
        crisp_inputs = [
            norm(suhu, 'Tavg'), norm(kelembapan, 'RH_avg'), 
            norm(hujan, 'RR'), norm(cahaya, 'ss'), norm(angin, 'ff_avg')
        ]
        # Cegah diluar batas
        crisp_inputs = [max(0.0, min(1.0, x)) for x in crisp_inputs]
        
        f_vals = fuzzy_sys.fuzzify(crisp_inputs)
        agg = fuzzy_sys.evaluate_rules(f_vals)
        hasil_mamdani = fuzzy_sys.defuzzify_mamdani(agg)
        hasil_sugeno = fuzzy_sys.defuzzify_sugeno(agg)
        
        st.subheader("Hasil Inferensi Sistem:")
        c1, c2 = st.columns(2)
        c1.metric(label="Indeks Kekeringan Mamdani", value=f"{hasil_mamdani:.2f}/100")
        c2.metric(label="Indeks Kekeringan Sugeno", value=f"{hasil_sugeno:.2f}/100")
        
        plot_fuzzy_results(f_vals, agg, hasil_mamdani, fuzzy_sys)

elif menu == "Integrasi Deep Learning (Forecasting)":
    st.markdown("<h1 class='highlight'>Integrasi Deep Learning + Fuzzy Logic</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <p>Bagian ini mendemonstrasikan arsitektur <b>Hybrid Forecasting-Inference</b> secara langsung.</p>
    <ul>
        <li><b>Deep Learning (MLP Neural Network)</b> digunakan untuk memprediksi/forecast fitur cuaca masa depan berdasarkan deret waktu historis (3 hari terakhir).</li>
        <li><b>Fuzzy Logic</b> menerima parameter cuaca yang diprediksi oleh Deep Learning untuk memutuskan <b>Indeks Kekeringan</b>.</li>
    </ul>
    <i>*Memenuhi syarat penting: Deep Learning tidak menggantikan Fuzzy Logic, melainkan berintegrasi di tahap prapemrosesan (forecasting).</i>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Data Input Observasi (3 Hari Terakhir)")
    recent_data = df[['Tavg', 'RH_avg', 'RR', 'ss', 'ff_avg']].tail(3)
    st.dataframe(recent_data, use_container_width=True)
    
    st.markdown("---")
    if st.button("Jalankan Prediksi Hybrid (DL -> Fuzzy)"):
        try:
            model = joblib.load('dl_model.pkl')
            scaler = joblib.load('scaler.pkl')
            
            # Siapkan input sequence (3 hari)
            data_scaled = scaler.transform(recent_data.values)
            X_input = data_scaled.flatten().reshape(1, -1)
            
            # Deep Learning Prediction
            y_pred_scaled = model.predict(X_input)
            
            # Memastikan dimensi y_pred_scaled sesuai (1, 5) karena sklearn MLP kadang outputnya 1D
            if len(y_pred_scaled.shape) == 1:
                y_pred_scaled = y_pred_scaled.reshape(1, -1)
                
            y_pred = scaler.inverse_transform(y_pred_scaled)
            
            st.success("✅ Deep Learning Model (Neural Network) berhasil memprediksi parameter cuaca untuk esok hari!")
            
            st.subheader("Hasil Prediksi Cuaca Esok Hari (Oleh Deep Learning)")
            pred_df = pd.DataFrame(y_pred, columns=['Suhu (Tavg)', 'Kelembapan (RH_avg)', 'Hujan (RR)', 'Cahaya (ss)', 'Angin (ff_avg)'])
            st.dataframe(pred_df, use_container_width=True)
            
            st.markdown("---")
            st.info("🔄 Meneruskan parameter prediksi ke Sistem Inferensi Fuzzy...")
            
            # Feed to Fuzzy Logic
            def norm(val, col):
                return (val - df[col].min()) / (df[col].max() - df[col].min())
                
            crisp_inputs = [
                norm(y_pred[0][0], 'Tavg'), norm(y_pred[0][1], 'RH_avg'), 
                norm(y_pred[0][2], 'RR'), norm(y_pred[0][3], 'ss'), norm(y_pred[0][4], 'ff_avg')
            ]
            
            # Bounding normalisasi agar tidak fail membership function
            crisp_inputs = [max(0.0, min(1.0, x)) for x in crisp_inputs]
            
            f_vals = fuzzy_sys.fuzzify(crisp_inputs)
            agg = fuzzy_sys.evaluate_rules(f_vals)
            hasil_mamdani = fuzzy_sys.defuzzify_mamdani(agg)
            hasil_sugeno = fuzzy_sys.defuzzify_sugeno(agg)
            
            st.subheader("Hasil Akhir Indeks Kekeringan (Oleh Fuzzy Logic)")
            c1, c2 = st.columns(2)
            c1.metric(label="Indeks Mamdani (Prediksi Esok)", value=f"{hasil_mamdani:.2f} / 100")
            c2.metric(label="Indeks Sugeno (Prediksi Esok)", value=f"{hasil_sugeno:.2f} / 100")
            
            plot_fuzzy_results(f_vals, agg, hasil_mamdani, fuzzy_sys)
            
        except FileNotFoundError:
            st.error("❌ Model Deep Learning tidak ditemukan! Silakan jalankan `train_dl.py` terlebih dahulu untuk melatih dan menghasilkan file `dl_model.pkl`.")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memprediksi: {e}")
