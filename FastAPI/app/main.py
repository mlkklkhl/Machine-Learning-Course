import uvicorn
from fastapi import FastAPI, Body
from joblib import load
import pickle

app = FastAPI(title="Epidemiological Disease Detection API",
              description="API for Epidemiological Disease Detection",
              version="1.0")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Epidemiological Disease Detection API!"}

@app.post('/prediction', tags=["predictions"])
async def get_prediction(Disease, Fever, Cough, Fatigue, DifficultyBreathing, Age, Gender, BloodPressure,
                         CholesterolLevel):

    # Load model from best_model.pkl
    model = load('../../models/best_model.pkl')
    # Load mapping from models/mapping.pkl
    mapping = load('../../models/mapping.pkl')
    # Load columns from models/columns.pkl
    columns = load('../../models/columns.pkl')

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
