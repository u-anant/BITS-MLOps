import pickle

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(data: IrisData):
    input_data = np.array(
        [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    )
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris prediction API!"}
