import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

def main():
    # 1) Crear carpeta de salida
    os.makedirs("data/processed", exist_ok=True)

    # 2) Cargar datos brutos
    df = pd.read_csv("data/raw/raw.csv")

    # 3) Conservar solo columnas numéricas
    df_numeric = df.select_dtypes(include=["number"])

    # 4) Separar variables predictoras y objetivo
    X = df_numeric.iloc[:, :-1]
    y = df_numeric.iloc[:, -1]

    # 5) División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6) Guardar particiones
    joblib.dump(X_train, "data/processed/X_train.pkl")
    joblib.dump(X_test,  "data/processed/X_test.pkl")
    joblib.dump(y_train, "data/processed/y_train.pkl")
    joblib.dump(y_test,  "data/processed/y_test.pkl")

if __name__ == "__main__":
    main()


