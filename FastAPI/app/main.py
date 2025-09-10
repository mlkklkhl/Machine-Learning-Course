import uvicorn
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import pickle
from pydantic import BaseModel
import logging

app = FastAPI(title="Epidemiological Disease Detection API",
              description="API for Epidemiological Disease Detection",
              version="1.0")

# Add CORS middleware - THIS IS CRITICAL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # This allows OPTIONS, GET, POST, etc.
    allow_headers=["*"],  # This allows all headers
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Epidemiological Disease Detection API!"}

@app.post('/prediction', tags=["predictions"])
async def get_prediction(Disease, Fever, Cough, Fatigue, DifficultyBreathing, Age, Gender, BloodPressure,
                         CholesterolLevel):

    # Load model from best_model.pkl
    model = load('../models/best_model.pkl')
    # Load mapping from models/mapping.pkl
    mapping = load('../models/mapping.pkl')
    # Load columns from models/columns.pkl
    columns = load('../models/columns.pkl')

    conditions = [Disease, Fever, Cough, Fatigue, DifficultyBreathing, Age, Gender, BloodPressure, CholesterolLevel]
    # map the conditions to the model columns
    data = []
    # convert test data to numerical using mapping from label encoder
    for i in columns:
        print(i, conditions[columns.get_loc(i)])
        if i == 'Age':
            data.append(int(conditions[columns.get_loc(i)]))
        else:
            data.append(list(mapping[i].values())[list(mapping[i].keys()).index(conditions[columns.get_loc(i)])])

    print(data)

    # prediction = 'test'
    prediction = model.predict([data]).tolist()
    print(prediction)
    prediction = list(mapping['Outcome Variable'].keys())[list(mapping['Outcome Variable'].values()).index(prediction)]

    return { "prediction": prediction }


class PredictionInput(BaseModel):
    Disease: str
    Fever: str
    Cough: str
    Fatigue: str
    DifficultyBreathing: str
    Age: int
    Gender: str
    BloodPressure: str
    CholesterolLevel: str

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.post('/prediction_web', tags=["predictions"])
async def get_prediction(request: Request, input_data: PredictionInput = Body(...)):
    try:
        # Load model from best_model.pkl
        model = load('../models/best_model.pkl')
        # Load mapping from models/mapping.pkl
        mapping = load('../models/mapping.pkl')
        # Load columns from models/columns.pkl
        columns = load('../models/columns.pkl')

        logger.debug(f"Received prediction request: {await request.json()}")

        if model is None or mapping is None or columns is None:
            raise HTTPException(status_code=503, detail="Models not loaded properly")

        input_dict = input_data.dict()
        logger.debug(f"Input dict: {input_dict}")

        # Extract conditions in the order of columns if column is DifficultyBreathing, BloodPressure, CholesterolLevel, add spaces in between
        if input_dict.get('DifficultyBreathing'):
            input_dict['Difficulty Breathing'] = input_dict.pop('DifficultyBreathing')
        if input_dict.get('BloodPressure'):
            input_dict['Blood Pressure'] = input_dict.pop('BloodPressure')
        if input_dict.get('CholesterolLevel'):
            input_dict['Cholesterol Level'] = input_dict.pop('CholesterolLevel')

        conditions = [input_dict[col] for col in columns]
        logger.debug(f"Conditions: {conditions}")

        # map the conditions to the model columns
        data = []
        # convert test data to numerical using mapping from label encoder
        for i in columns:
            print(i, conditions[columns.get_loc(i)])
            if i == 'Age':
                data.append(int(conditions[columns.get_loc(i)]))
            else:
                data.append(list(mapping[i].values())[list(mapping[i].keys()).index(conditions[columns.get_loc(i)])])

        logger.debug(f"Processed data: {data}")

        prediction_numeric = model.predict([data])[0]
        logger.debug(f"Raw prediction: {prediction_numeric}")

        if 'Outcome Variable' in mapping:
            reverse_mapping = {v: k for k, v in mapping['Outcome Variable'].items()}
            prediction = reverse_mapping.get(prediction_numeric, "Unknown")
        else:
            prediction = str(prediction_numeric)

        response_data = {
            "prediction": prediction,
            "status": "success"
        }

        logger.debug(f"Response: {response_data}")
        return response_data

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
