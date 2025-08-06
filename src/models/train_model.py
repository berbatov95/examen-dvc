import os
import joblib
from sklearn.ensemble import RandomForestRegressor

def main():
    # 1) Crear carpeta de salida
    os.makedirs("models/models", exist_ok=True)

    # 2) Cargar datos escalados y mejores parámetros
    X_train    = joblib.load("data/processed/X_train_scaled.pkl")
    y_train    = joblib.load("data/processed/y_train.pkl")
    best_params = joblib.load("models/grid/best_params.pkl")

    # 3) Entrenar el modelo
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # 4) Guardar el modelo entrenado
    joblib.dump(model, "models/models/final_model.pkl")

if __name__ == "__main__":
    main()
