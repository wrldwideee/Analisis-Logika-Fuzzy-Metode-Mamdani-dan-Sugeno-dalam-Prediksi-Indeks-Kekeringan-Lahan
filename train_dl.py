import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    cols = ['Tavg', 'RH_avg', 'RR', 'ss', 'ff_avg']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.ffill().bfill()
    data = df[cols].values

    print("Normalisasi data...")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    look_back = 3
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:(i + look_back)].flatten())
        y.append(data_scaled[i + look_back])

    X = np.array(X)
    y = np.array(y)

    print("Membangun dan melatih model Deep Learning...")
    model = MLPRegressor(hidden_layer_sizes=(64, 32), 
                         activation='relu',
                         solver='adam',
                         max_iter=300, 
                         random_state=42,
                         verbose=True)
    model.fit(X, y)

    # === HITUNG METRIK ASLI ===
    print("Menghitung metrik performa...")
    y_pred_all = model.predict(X)
    r2 = r2_score(y, y_pred_all)
    mae = mean_absolute_error(y, y_pred_all)
    mse = mean_squared_error(y, y_pred_all)
    
    metrics = {'r2': r2, 'mae': mae, 'mse': mse}

    # === SIMPAN SEMUA FILE KE FOLDER ===
    print("Menyimpan model, scaler, dan metrics...")
    joblib.dump(model, 'dl_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(metrics, 'metrics.pkl') # <-- Wajib ada ini!

    print("Selesai! Semua file berhasil dibuat.")

if __name__ == "__main__":
    main()