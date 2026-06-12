import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Plot fungsi keanggotaan trapesium untuk semua variabel input
# ─────────────────────────────────────────────────────────────────────────────
def plot_membership_functions(crisp_inputs, title_prefix=""):
    """
    Menampilkan 5 subplot fungsi keanggotaan trapesium (satu per variabel input)
    beserta garis vertikal posisi nilai crisp saat ini.
    """
    var_names  = ['Suhu', 'Kelembapan', 'Hujan', 'Cahaya', 'Angin']
    var_labels = ['Suhu (Tavg)', 'Kelembapan (RH_avg)', 'Hujan (RR)',
                  'Lama Cahaya (ss)', 'Kecepatan Angin (ff_avg)']
    colors     = {'Rendah': '#8fd3f4', 'Sedang': '#84fab0', 'Tinggi': '#ff6b6b'}
    x_range    = fuzzy_sys.x_range
    params     = fuzzy_sys.input_trap_params

    st.markdown(f"**{title_prefix} — Fungsi Keanggotaan Trapesium (5 Variabel Input)**")

    fig, axes = plt.subplots(1, 5, figsize=(20, 3.5), sharey=True)
    fig.patch.set_facecolor('#0E1117')

    for ax, var, label, crisp_val in zip(axes, var_names, var_labels, crisp_inputs):
        ax.set_facecolor('#1E2127')
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

        for term, color in colors.items():
            mu = fuzzy_sys.trapmf(x_range, params[term])
            ax.plot(x_range, mu, color=color, linewidth=2, label=term)
            ax.fill_between(x_range, 0, mu, color=color, alpha=0.15)

        # Garis vertikal nilai crisp
        ax.axvline(x=crisp_val, color='yellow', linewidth=1.8,
                   linestyle='--', label=f'Input: {crisp_val:.2f}')

        # Titik derajat keanggotaan pada nilai crisp
        for term, color in colors.items():
            mu_val = fuzzy_sys.trapmf(crisp_val, params[term])
            ax.scatter([crisp_val], [mu_val], color=color, s=60, zorder=5)

        ax.set_title(label, color='white', fontsize=9, pad=6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel('Nilai Ternormalisasi', color='#aaa', fontsize=7)

    axes[0].set_ylabel('Derajat Keanggotaan (μ)', color='#aaa', fontsize=8)

    # Legend bersama di luar plot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               facecolor='#1E2127', labelcolor='white', fontsize=9,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.18))

    fig.suptitle(f"{title_prefix} — Fungsi Keanggotaan Trapesium Input",
                 color='white', fontsize=11, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Plot fungsi keanggotaan output Mamdani + area defuzzifikasi
# ─────────────────────────────────────────────────────────────────────────────
def plot_mamdani_output(agg, hasil_mamdani):
    y_range    = fuzzy_sys.y_range
    out_params = fuzzy_sys.output_trap_params
    colors     = {'Rendah': '#8fd3f4', 'Sedang': '#84fab0', 'Tinggi': '#ff6b6b'}

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#1E2127')
    ax.set_facecolor('#1E2127')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    for term, color in colors.items():
        mu_full = fuzzy_sys.trapmf(y_range, out_params[term])
        ax.plot(y_range, mu_full, color=color, linestyle='--', alpha=0.55, linewidth=1.5, label=term)

    shaded_y = np.array([
        max(
            min(agg['Rendah'], fuzzy_sys.trapmf(y, out_params['Rendah'])),
            min(agg['Sedang'], fuzzy_sys.trapmf(y, out_params['Sedang'])),
            min(agg['Tinggi'], fuzzy_sys.trapmf(y, out_params['Tinggi']))
        ) for y in y_range
    ])

    ax.fill_between(y_range, 0, shaded_y, color='#a777e3', alpha=0.75, label='Area Agregasi')
    ax.axvline(x=hasil_mamdani, color='white', linestyle='-', linewidth=2.0,
               label=f'Centroid: {hasil_mamdani:.1f}')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('Indeks Kekeringan', color='#aaa', fontsize=9)
    ax.set_ylabel('μ', color='#aaa', fontsize=9)
    ax.set_title('Area Defuzzifikasi Mamdani (Trapesium)', color='white', fontsize=10)

    legend = ax.legend(loc='upper right', fontsize='small', facecolor='#1E2127',
                       framealpha=0.8)
    for text in legend.get_texts():
        text.set_color('white')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Heatmap fuzzifikasi
# ─────────────────────────────────────────────────────────────────────────────
def plot_heatmap(f_vals):
    labels = ['Suhu', 'Kelembapan', 'Hujan', 'Cahaya', 'Angin']
    terms  = ['Rendah', 'Sedang', 'Tinggi']
    data   = [[f_vals[lbl][t] for t in terms] for lbl in labels]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#1E2127')
    ax.set_facecolor('#1E2127')
    sns.heatmap(data, annot=True, fmt='.2f', cmap='magma',
                xticklabels=terms, yticklabels=labels, ax=ax, vmin=0, vmax=1)
    ax.tick_params(colors='white')
    ax.set_title('Derajat Keanggotaan (Fuzzifikasi)', color='white', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Scatter plot Mamdani vs Sugeno (semua data dataset)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def compute_all_mamdani_sugeno():
    """Menghitung indeks Mamdani & Sugeno untuk seluruh baris dataset."""
    cols = ['Tavg', 'RH_avg', 'RR', 'ss', 'ff_avg']
    data = df[cols].copy()

    mamdani_vals, sugeno_vals = [], []

    for col in cols:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max == col_min:
            data[col] = 0.5
        else:
            data[col] = (data[col] - col_min) / (col_max - col_min)

    data = data.clip(0.0, 1.0)

    fs = FuzzyKekeringan()
    for _, row in data.iterrows():
        crisp = row[cols].tolist()
        fv    = fs.fuzzify(crisp)
        agg   = fs.evaluate_rules(fv)
        mamdani_vals.append(fs.defuzzify_mamdani(agg))
        sugeno_vals.append(fs.defuzzify_sugeno(agg))

    return np.array(mamdani_vals), np.array(sugeno_vals)


def plot_mamdani_vs_sugeno(highlight_mamdani=None, highlight_sugeno=None):
    """
    Scatter plot perbandingan output Mamdani vs Sugeno untuk seluruh dataset,
    dengan titik highlight untuk nilai saat ini (opsional).
    """
    mamdani_vals, sugeno_vals = compute_all_mamdani_sugeno()

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#1E2127')
    ax.set_facecolor('#1E2127')
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')
    ax.grid(True, color='#333', linestyle='--', linewidth=0.6)

    # Diagonal referensi y = x
    ax.plot([0, 100], [0, 100], color='#ff4444', linestyle='--',
            linewidth=1.8, label='Garis y=x (referensi)')

    # Scatter seluruh data
    ax.scatter(mamdani_vals, sugeno_vals, color='#6e8efb', alpha=0.6,
               edgecolors='none', s=35, label='Semua Data')

    # Titik highlight (nilai saat ini)
    if highlight_mamdani is not None and highlight_sugeno is not None:
        ax.scatter([highlight_mamdani], [highlight_sugeno],
                   color='yellow', edgecolors='white', s=150, zorder=10,
                   linewidths=1.5, label=f'Input saat ini\n(M:{highlight_mamdani:.1f}, S:{highlight_sugeno:.1f})')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Output Indeks Kekeringan (Mamdani)', color='white', fontsize=10)
    ax.set_ylabel('Output Indeks Kekeringan (Sugeno)',  color='white', fontsize=10)
    ax.set_title('Perbandingan Output: Mamdani vs Sugeno (Seluruh Data)',
                 color='white', fontsize=11, pad=10)

    legend = ax.legend(loc='upper left', fontsize='small',
                       facecolor='#1E2127', framealpha=0.85)
    for text in legend.get_texts():
        text.set_color('white')

    # Korelasi
    if len(mamdani_vals) > 1:
        corr = np.corrcoef(mamdani_vals, sugeno_vals)[0, 1]
        ax.text(0.98, 0.05, f'r = {corr:.4f}', transform=ax.transAxes,
                ha='right', color='#84fab0', fontsize=10,
                bbox=dict(facecolor='#0E1117', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Tampilan hasil fuzzy lengkap (heatmap + MF output + scatter)
# ─────────────────────────────────────────────────────────────────────────────
def show_full_fuzzy_results(crisp_inputs, f_vals, agg,
                             hasil_mamdani, hasil_sugeno, title_prefix=""):
    st.markdown("---")
    st.markdown(f"<h3 class='highlight'>Visualisasi Keputusan Fuzzy — {title_prefix}</h3>",
                unsafe_allow_html=True)

    # ── Baris 1: Fungsi Keanggotaan Input Trapesium ──
    plot_membership_functions(crisp_inputs, title_prefix)

    st.markdown("---")

    # ── Baris 2: Heatmap + Area Mamdani ──
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Heatmap Fuzzifikasi (Derajat Keanggotaan)**")
        plot_heatmap(f_vals)
    with colB:
        st.markdown("**Area Defuzzifikasi Mamdani**")
        plot_mamdani_output(agg, hasil_mamdani)

    st.markdown("---")

    # ── Baris 3: Scatter Mamdani vs Sugeno ──
    st.markdown("**Perbandingan Output Mamdani vs Sugeno (Seluruh Dataset)**")
    plot_mamdani_vs_sugeno(
        highlight_mamdani=hasil_mamdani,
        highlight_sugeno=hasil_sugeno
    )


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGASI
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# HALAMAN 1: Beranda & Dataset
# ─────────────────────────────────────────────────────────────────────────────
if menu == "Beranda & Dataset":
    st.markdown("<h1 class='highlight'>Analisis Tingkat Kekeringan Harian BMKG</h1>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h3>Selamat Datang di Sistem Prediksi Kekeringan</h3>
        <p>Aplikasi ini memadukan <b>Logika Fuzzy</b> murni (Mamdani & Sugeno) dengan
        <b>Deep Learning</b> (Multi-Layer Perceptron Neural Network) untuk memprediksi
        tingkat kekeringan berdasarkan data historis stasiun iklim harian BMKG.</p>
        <p>Fungsi keanggotaan yang digunakan adalah <b>Fungsi Trapesium</b> untuk kelima
        variabel input: Suhu, Kelembapan, Curah Hujan, Lama Penyinaran, dan Kecepatan Angin.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Cuplikan Data Iklim Historis (100 Terakhir)")
    st.dataframe(df.tail(100), use_container_width=True)

    st.subheader("Tren Suhu & Kelembapan")
    chart_data = df[['Tavg', 'RH_avg']].tail(100)
    st.line_chart(chart_data)

# ─────────────────────────────────────────────────────────────────────────────
# HALAMAN 2: Sistem Fuzzy Manual
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "Sistem Fuzzy (Manual)":
    st.markdown("<h1 class='highlight'>Simulasi Fuzzy Kekeringan</h1>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>Sesuaikan parameter iklim pada slider di bawah untuk melihat bagaimana
        Logika Fuzzy memproses nilai crisp menjadi Indeks Kekeringan menggunakan
        <b>15 rule</b> dan <b>fungsi keanggotaan trapesium</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        suhu      = st.slider("Suhu Rata-rata (Tavg)",
                              float(df['Tavg'].min()), float(df['Tavg'].max()),
                              float(df['Tavg'].mean()))
        kelembapan = st.slider("Kelembapan Rata-rata (RH_avg)",
                               float(df['RH_avg'].min()), float(df['RH_avg'].max()),
                               float(df['RH_avg'].mean()))
    with col2:
        hujan  = st.slider("Curah Hujan (RR)",
                           float(df['RR'].min()), float(df['RR'].max()),
                           float(df['RR'].mean()))
        cahaya = st.slider("Lama Penyinaran (ss)",
                           float(df['ss'].min()), float(df['ss'].max()),
                           float(df['ss'].mean()))
    with col3:
        angin = st.slider("Kecepatan Angin (ff_avg)",
                          float(df['ff_avg'].min()), float(df['ff_avg'].max()),
                          float(df['ff_avg'].mean()))

    st.markdown("---")
    if st.button("Hitung Tingkat Kekeringan (Inferensi Fuzzy)"):
        def norm(val, col):
            col_min, col_max = df[col].min(), df[col].max()
            if col_max == col_min:
                return 0.5
            return (val - col_min) / (col_max - col_min)

        crisp_inputs = [
            norm(suhu, 'Tavg'), norm(kelembapan, 'RH_avg'),
            norm(hujan, 'RR'), norm(cahaya, 'ss'), norm(angin, 'ff_avg')
        ]
        crisp_inputs = [max(0.0, min(1.0, x)) for x in crisp_inputs]

        f_vals       = fuzzy_sys.fuzzify(crisp_inputs)
        agg          = fuzzy_sys.evaluate_rules(f_vals)
        hasil_mamdani = fuzzy_sys.defuzzify_mamdani(agg)
        hasil_sugeno  = fuzzy_sys.defuzzify_sugeno(agg)

        st.subheader("Hasil Inferensi Sistem:")
        c1, c2 = st.columns(2)
        c1.metric(label="Indeks Kekeringan Mamdani", value=f"{hasil_mamdani:.2f}/100")
        c2.metric(label="Indeks Kekeringan Sugeno",  value=f"{hasil_sugeno:.2f}/100")

        show_full_fuzzy_results(
            crisp_inputs, f_vals, agg,
            hasil_mamdani, hasil_sugeno,
            title_prefix="Fuzzy Manual"
        )

# ─────────────────────────────────────────────────────────────────────────────
# HALAMAN 3: Integrasi Deep Learning
# ─────────────────────────────────────────────────────────────────────────────
elif menu == "Integrasi Deep Learning (Forecasting)":
    st.markdown("<h1 class='highlight'>Integrasi Deep Learning + Fuzzy Logic</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <p>Bagian ini mendemonstrasikan arsitektur <b>Hybrid Forecasting-Inference</b> secara langsung.</p>
    <ul>
        <li><b>Deep Learning (MLP Neural Network)</b> memprediksi fitur cuaca masa depan berdasarkan 3 hari terakhir.</li>
        <li><b>Fuzzy Logic</b> menerima parameter prediksi tersebut dan memutuskan <b>Indeks Kekeringan</b>.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📋 Data Input Observasi (3 Hari Terakhir)")
    recent_data = df[['Tavg', 'RH_avg', 'RR', 'ss', 'ff_avg']].tail(3)
    st.dataframe(recent_data, use_container_width=True)

    st.markdown("---")
    
    if st.button("Jalankan Prediksi Hybrid (DL → Fuzzy)"):
        try:
            # 1. LOAD MODEL, SCALER, DAN METRIK HASIL TRAINING
            model  = joblib.load('dl_model.pkl')
            scaler = joblib.load('scaler.pkl')
            saved_metrics = joblib.load('metrics.pkl')

            # 2. PROSES PREDIKSI OLEH DEEP LEARNING
            data_scaled = scaler.transform(recent_data.values)
            X_input     = data_scaled.flatten().reshape(1, -1)
            
            y_pred_scaled = model.predict(X_input)
            if len(y_pred_scaled.shape) == 1:
                y_pred_scaled = y_pred_scaled.reshape(1, -1)
            y_pred = scaler.inverse_transform(y_pred_scaled)

            st.success("✅ Proses Kalkulasi Selesai! Model Deep Learning dan Fuzzy Berhasil Terintegrasi.")
            st.markdown("---")

            # === OUTPUT 1: METRIK PERFORMA MODEL DL ===
            st.subheader("📊 1. Performa Model Deep Learning (MLP Regressor)")
            st.caption("Metrik evaluasi regresi runtun waktu berdasarkan hasil training data nyata:")
            
            # Membuat 5 kolom horizontal
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            
            col_m1.metric(label="R-Squared (R²)", value=f"{saved_metrics['r2']:.3f}")
            col_m2.metric(label="MAE", value=f"{saved_metrics['mae']:.4f}")
            col_m3.metric(label="MSE", value=f"{saved_metrics['mse']:.4f}")
            col_m4.metric(label="RMSE", value=f"{saved_metrics['rmse']:.4f}")
            col_m5.metric(label="MAPE", value=f"{saved_metrics['mape']:.2f}%")

            st.markdown("---")

            # === OUTPUT 2: HASIL PREDIKSI FITUR CUACA ===
            st.subheader("🔮 2. Hasil Prediksi Parameter Cuaca Esok Hari")
            pred_df = pd.DataFrame(
                y_pred,
                columns=['Suhu (Tavg)', 'Kelembapan (RH_avg)', 'Hujan (RR)', 'Cahaya (ss)', 'Angin (ff_avg)']
            )
            st.dataframe(pred_df, use_container_width=True)

            # === OUTPUT 3: GRAFIK KOMPARASI AKTUAL VS PREDIKSI ===
            st.subheader("📈 3. Komparasi Fitur: Aktual Hari Ini vs Prediksi Esok Hari")
            
            aktual_terakhir = recent_data.iloc[-1].values
            fitur_names = ['Suhu (Tavg)', 'Kelembapan (RH_avg)', 'Hujan (RR)', 'Cahaya (ss)', 'Angin (ff_avg)']
            
            fig_comp, ax_comp = plt.subplots(figsize=(8, 3.5))
            fig_comp.patch.set_facecolor('#1E2127')
            ax_comp.set_facecolor('#1E2127')
            
            x_indices = np.arange(len(fitur_names))
            bar_width = 0.35
            
            ax_comp.bar(x_indices - bar_width/2, aktual_terakhir, bar_width, label='Aktual (Hari Ini)', color='#6e8efb')
            ax_comp.bar(x_indices + bar_width/2, y_pred[0], bar_width, label='Prediksi (Esok Hari)', color='#84fab0')
            
            ax_comp.set_xticks(x_indices)
            ax_comp.set_xticklabels(fitur_names, color='white', fontsize=8)
            ax_comp.tick_params(colors='white')
            ax_comp.legend(facecolor='#1E2127', labelcolor='white')
            ax_comp.set_title("Perbandingan Nilai Fitur Cuaca", color='white', fontsize=10)
            for spine in ax_comp.spines.values():
                spine.set_edgecolor('#444')
                
            st.pyplot(fig_comp)
            plt.close(fig_comp)

            # === OUTPUT 4: PROSES KE FUZZY LOGIC ===
            st.markdown("---")
            st.info("🔄 Meneruskan parameter hasil prediksi Deep Learning ke Sistem Inferensi Fuzzy...")

            def norm(val, col):
                col_min, col_max = df[col].min(), df[col].max()
                return (val - col_min) / (col_max - col_min) if col_max != col_min else 0.5

            crisp_inputs = [max(0.0, min(1.0, norm(y_pred[0][i], col))) for i, col in enumerate(['Tavg', 'RH_avg', 'RR', 'ss', 'ff_avg'])]

            f_vals        = fuzzy_sys.fuzzify(crisp_inputs)
            agg           = fuzzy_sys.evaluate_rules(f_vals)
            hasil_mamdani = fuzzy_sys.defuzzify_mamdani(agg)
            hasil_sugeno  = fuzzy_sys.defuzzify_sugeno(agg)

            st.subheader("🎯 4. Hasil Akhir Indeks Kekeringan (Oleh Fuzzy Logic)")
            c1, c2 = st.columns(2)
            c1.metric(label="Indeks Mamdani (Prediksi Esok)", value=f"{hasil_mamdani:.2f} / 100")
            c2.metric(label="Indeks Sugeno (Prediksi Esok)",  value=f"{hasil_sugeno:.2f} / 100")

            # === MODIFIKASI: Memecah isi show_full_fuzzy_results tanpa memanggil plot_mamdani_vs_sugeno ===
            st.markdown("---")
            st.markdown("<h3 class='highlight'>Visualisasi Keputusan Fuzzy — DL + Fuzzy</h3>", unsafe_allow_html=True)
            
            # Tampilkan Grafik Batang Fungsi Keanggotaan Trapesium Input
            plot_membership_functions(crisp_inputs, "DL + Fuzzy")
            st.markdown("---")
            
            # Hanya tampilkan Heatmap dan Area Defuzzifikasi Centroid Mamdani
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Heatmap Fuzzifikasi (Derajat Keanggotaan)**")
                plot_heatmap(f_vals)
            with colB:
                st.markdown("**Area Defuzzifikasi Mamdani**")
                plot_mamdani_output(agg, hasil_mamdani)

        except FileNotFoundError:
            st.error("❌ Model Deep Learning atau file metrik tidak ditemukan! Silakan jalankan `train_dl.py` terlebih dahulu untuk men-generate file pkl.")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")