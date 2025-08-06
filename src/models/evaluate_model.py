import os
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # 1) Crear carpeta de salida si no existe
    os.makedirs("metrics", exist_ok=True)

    # 2) Cargar datos de test y modelo
    X_test = joblib.load("data/processed/X_test_scaled.pkl")
    y_test = joblib.load("data/processed/y_test.pkl")
    model  = joblib.load("models/models/final_model.pkl")

    # 3) Hacer predicciones
    y_pred = model.predict(X_test)

    # 4) Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test,  y_pred)
    results = {"mse": mse, "r2": r2}

    # 5) Guardar métricas en JSON
    with open("metrics/scores.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
