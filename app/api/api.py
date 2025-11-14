import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient
from dotenv import load_dotenv
import os

load_dotenv(override=True)  # Carga las variables del archivo .env

mlflow.set_tracking_uri("databricks")
client = MlflowClient()

EXPERIMENT_NAME = "/Users/isabel.valladolid@iteso.mx/nyc-taxi-experiments"

# search for best run
run_ = mlflow.search_runs(order_by=['metrics.rmse ASC'],
                          output_format="list",
                          experiment_names=[EXPERIMENT_NAME]
                          )[0]

# search for runs IDs
run_id = run_.info.run_id
run_uri = f"runs:/{run_id}/preprocessor"

# download preprocessor artifact
client.download_artifacts(
    run_id=run_id,
    path='preprocessor',
    dst_path='.'
)

# load preprocessor
with open("preprocessor/preprocessor.b", "rb") as f_in:
    dv = pickle.load(f_in)

# select model
model_name = "workspace.default.nyc-taxi-model"
alias = "champion"
model_uri = f"models:/{model_name}@{alias}"

# load model
champion_model = mlflow.pyfunc.load_model(
    model_uri=model_uri
)

# preprocessing function
def preprocess(input_data):

    input_dict = {
        'PU_DO': input_data.PULocationID + "_" + input_data.DOLocationID,
        'trip_distance': input_data.trip_distance,
    }

    return dv.transform(input_dict)

# prediction function
def predict(input_data):

    X_val = preprocess(input_data)

    return "4"

# API
app = FastAPI()

class InputData(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float


@app.post("/api/v1/predict")
def predict_endpoint(input_data: InputData):
    result = predict(input_data)[0]
    return {"prediction": float(result)}

