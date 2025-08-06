import os
import joblib
from sklearn.preprocessing import StandardScaler

def main():
    # 1) Crear carpetas de salida si no existen
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/data",    exist_ok=True)

    # 2) Cargar los datos ya partidos
    X_train = joblib.load("data/processed/X_train.pkl")
    X_test  = joblib.load("data/processed/X_test.pkl")

    # 3) Ajustar scaler y transformar
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 4) Volcar scaler y datos escalados
    joblib.dump(scaler,             "models/data/scaler.pkl")
    joblib.dump(X_train_scaled,     "data/processed/X_train_scaled.pkl")
    joblib.dump(X_test_scaled,      "data/processed/X_test_scaled.pkl")

if __name__ == "__main__":
    main()
