import pickle
import logging
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise e

app = FastAPI()


class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(data: IrisData):
    input_data = np.array([[data.sepal_length, data.sepal_width,
                            data.petal_length, data.petal_width]])
    logger.info(f"Received data: {data}")
    prediction = model.predict(input_data)
    logger.info(f"Prediction made: {prediction[0]}")
    return {"prediction": int(prediction[0])}


@app.get("/")
def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Iris prediction API!"}
