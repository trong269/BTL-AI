from fastapi import FastAPI
from pydantic import BaseModel

from service import Service
import pandas as pd


class PredictResponse(BaseModel):
    predicted_price: float

class PredictRequest(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

service = Service()
df = pd.read_csv(r'./data/housing.csv')
service.train(df=df, test_size=0.2, learning_rate=1e-3, epochs=10000)
    
@app.post ("/predict", response_model=PredictResponse)
async def predict(request : PredictRequest):
    data = request.dict()
    request_df = pd.DataFrame([data])
    prediction = service.inference(request_df)
    return PredictResponse(predicted_price=prediction[0][0])