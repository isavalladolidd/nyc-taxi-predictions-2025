from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
import pathlib
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import mlflow
from mlflow.tracking import MlflowClient

# Configs
MLFLOW_EXPERIMENT = "nyc-taxi-experiment-prefect"
MODEL_REGISTRY = "nyc-taxi-model-prefect"


@task
def load_data_task(path_hints=None):
    # Busca archivos de datos comunes dentro del repo (puedes editar las rutas)
    project_root = pathlib.Path(__file__).resolve().parents[1]
    candidates = []
    if path_hints:
        candidates.extend(path_hints)
    candidates.extend([
        project_root / "data" / "processed.csv",
        project_root / "data" / "processed.parquet",
        project_root / "data" / "cleaned.csv",
        project_root / "data" / "cleaned.parquet",
        project_root / "data" / "dataset.csv",
    ])
    for p in candidates:
        p = pathlib.Path(p)
        if p.exists():
            if p.suffix in [".csv"]:
                df = pd.read_csv(p)
            elif p.suffix in [".parquet"]:
                df = pd.read_parquet(p)
            else:
                continue
            return df
    raise FileNotFoundError(
        "No se encontró un archivo de datos en las rutas comunes. "
        "Ponga la ruta en path_hints o coloque el dataset en data/processed.csv"
    )


@task
def prepare_data(df, target_column="fare_amount", test_size=0.2, random_state=42):
    if target_column not in df.columns:
        raise KeyError(f"Columna target '{target_column}' no encontrada en el dataframe.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    # Convertir categóricos sencillos si existen (fallback)
    X = pd.get_dummies(X, drop_first=True)
    # Alinea columnas por si faltan en test/train posteriores
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


@task
def train_model_task(X_train, y_train, model_type="rf", random_state=42):
    # model_type: 'rf' o 'ridge'
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    elif model_type == "ridge":
        model = Ridge(random_state=random_state)
    else:
        raise ValueError("model_type debe ser 'rf' o 'ridge'")
    model.fit(X_train, y_train)
    return model


@task
def evaluate_model_task(model, X_test, y_test):
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)  # mayor es mejor
    return float(score)


def _mlflow_register_model(model, alias, registry_name=MODEL_REGISTRY):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        # Log model artefacto con MLflow (sklearn)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        # Registrar en el Model Registry
        mv = mlflow.register_model(model_uri=model_uri, name=registry_name)
        client = MlflowClient()
        # Espera a que la versión exista (simple retry)
        import time
        for _ in range(10):
            versions = client.get_latest_versions(registry_name)
            if any(v.version == mv.version for v in versions):
                break
            time.sleep(1)
        # Asigna tag 'alias' al model version para identificar champion/challenger
        client.set_model_version_tag(registry_name, mv.version, "alias", alias)
        return mv.version


@task
def register_model_task(model, alias, registry_name=MODEL_REGISTRY):
    try:
        version = _mlflow_register_model(model, alias=alias, registry_name=registry_name)
        return {"name": registry_name, "version": version, "alias": alias}
    except Exception as e:
        raise RuntimeError(f"Error registrando el modelo en MLflow: {e}")


@task
def load_champion_task(registry_name=MODEL_REGISTRY, alias="@champion"):
    client = MlflowClient()
    # Buscar model versions con tag alias == alias
    filter_str = f"name = '{registry_name}' and tags.alias = '{alias}'"
    results = client.search_model_versions(filter_str)
    if not results:
        # fallback: obtener última versión registrada
        latest = client.get_latest_versions(registry_name)
        if not latest:
            raise RuntimeError(f"No hay versiones registradas para {registry_name}")
        # usar la versión con mayor versión numérica
        chosen = sorted(latest, key=lambda v: int(v.version), reverse=True)[0]
    else:
        chosen = results[0]
    model_uri = f"models:/{registry_name}/{chosen.version}"
    try:
        loaded = mlflow.pyfunc.load_model(model_uri)
        return loaded
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo desde {model_uri}: {e}")


@flow(name="nyc-taxi-experiment-prefect-challenger")
def main_flow(data_path_hints: list | None = None, target_column: str = "fare_amount"):
    # Cargar datos
    df = load_data_task(path_hints=data_path_hints)
    X_train, X_test, y_train, y_test = prepare_data(df, target_column=target_column).result()

    # Entrenar dos modelos (challenger A y B)
    model_a = train_model_task(X_train, y_train, model_type="rf")
    model_b = train_model_task(X_train, y_train, model_type="ridge")

    # Evaluar ambos
    score_a = evaluate_model_task(model_a, X_test, y_test)
    score_b = evaluate_model_task(model_b, X_test, y_test)

    # Decidir challenger (elige el mejor entre a/b) y luego comparar con champion registrado
    if score_a > score_b:
        challenger_model = model_a
        challenger_score = score_a
        challenger_name = "@challenger"
    else:
        challenger_model = model_b
        challenger_score = score_b
        challenger_name = "@challenger"

    # Registrar challenger provisionalmente con alias @challenger
    reg_info = register_model_task(challenger_model, alias=challenger_name).result()

    # Cargar champion desde registry (si existe)
    try:
        champion_loaded = load_champion_task(registry_name=MODEL_REGISTRY, alias="@champion")
        champion_score = evaluate_model_task(champion_loaded, X_test, y_test)
    except Exception:
        champion_loaded = None
        champion_score = -np.inf

    # Comparar
    if challenger_score > champion_score:
        
        client = MlflowClient()
        
        existing = client.search_model_versions(f"name = '{MODEL_REGISTRY}' and tags.alias = '@champion'")
        for v in existing:
            client.set_model_version_tag(MODEL_REGISTRY, v.version, "alias", "@previous_champion")
        
        version = str(reg_info["version"])
        client.set_model_version_tag(MODEL_REGISTRY, version, "alias", "@champion")
        best = "@champion"
    else:
        best = "@champion" if champion_loaded is not None else "@challenger"

    create_markdown_artifact(
        f"Challenger score: {challenger_score}\nChampion score: {champion_score}\nBest alias: {best}"
    )

    return {"challenger_score": challenger_score, "champion_score": champion_score, "best": best}


if __name__ == "__main__":
    main_flow()