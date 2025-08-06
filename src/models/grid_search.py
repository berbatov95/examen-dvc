import os
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def main():
    # 1) Crear carpeta de salida
    os.makedirs("models/grid", exist_ok=True)

    # 2) Cargar datos escalados y objetivo de entrenamiento
    X_train = joblib.load("data/processed/X_train_scaled.pkl")
    y_train = joblib.load("data/processed/y_train.pkl")

    # 3) Definir modelo y rejilla de parámetros
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth":    [None, 10, 20],
    }

    # 4) Ejecutar GridSearchCV
    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring="r2"
    )
    grid.fit(X_train, y_train)

    # 5) Serializar resultados de cv_results_ a listas para JSON
    cv_results = {}
    for key, values in grid.cv_results_.items():
        try:
            # convertir numpy array a lista
            cv_results[key] = values.tolist()
        except AttributeError:
            cv_results[key] = values

    # 6) Guardar mejores parámetros y resultados
    joblib.dump(grid.best_params_, "models/grid/best_params.pkl")
    with open("models/grid/grid_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

if __name__ == "__main__":
    main()

