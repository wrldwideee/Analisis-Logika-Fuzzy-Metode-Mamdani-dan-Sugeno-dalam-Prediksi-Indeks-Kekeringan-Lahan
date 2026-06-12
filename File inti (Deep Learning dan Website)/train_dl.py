import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

def main():
    print("Memuat dataset climate_data.csv...")
    try:
        df = pd.read_csv('climate_data.csv')
    except Exception as e:
        print(f"Gagal membaca dataset: {e}")
        return

    # Preprocessing
    df = df.ffill().bfill()
    
    # Memastikan kolom numerik
    cols = ['Tavg', 'RH_avg', 'RR', 'ss', 'ff_avg']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.ffill().bfill()

    data = df[cols].values

    # Normalisasi Data
    print("Normalisasi data...")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Membuat Sequences (Memprediksi hari esok berdasarkan 3 hari sebelumnya)
    X, y = [], []
    look_back = 3
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:(i + look_back)].flatten())
        y.append(data_scaled[i + look_back])

    X = np.array(X)
    y = np.array(y)

    print("Membangun dan melatih model Deep Learning (Multi-Layer Perceptron)...")
    # Arsitektur Neural Network Feed-Forward
    model = MLPRegressor(hidden_layer_sizes=(64, 32), 
                         activation='relu',
                         solver='adam',
                         max_iter=300, 
                         random_state=42,
                         verbose=True)
    
    # Melatih model (Bisa memakan waktu beberapa saat tergantung CPU)
    model.fit(X, y)

    # Menyimpan model dan scaler agar bisa di-load di Streamlit
    print("Menyimpan model ke dl_model.pkl dan scaler ke scaler.pkl...")
    joblib.dump(model, 'dl_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("Selesai! Model siap digunakan.")

if __name__ == "__main__":
    main()
