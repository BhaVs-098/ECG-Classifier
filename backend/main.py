from typing import List
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from scipy.signal import resample
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# # Load the trained model
model = joblib.load("ecg_random_forest.pkl")

class ECGInput(BaseModel):
    data: List[float]  # Ensure it expects a list of floats

@app.get("/")
def home():
    return {"message": "ECG Classifier API is Running!"}

# @app.post("/predict/")
# def predict_ecg(data: list):
#     try:
#         # Convert input data to numpy array
#         input_data = np.array(data).reshape(1, -1)
        
#         # Make prediction
#         prediction = model.predict(input_data)
        
#         return {"Predicted Class": int(prediction[0])}
#     except Exception as e:
#         return {"error": str(e)}



# @app.post("/predict/")
# async def predict_ecg(input_data: ECGInput):
#     # Convert input list to numpy array
#     ecg_array = np.array(input_data.data).reshape(1, -1)
#     prediction = model.predict(ecg_array)  # Run the prediction
#     return {"prediction": prediction.tolist()}



# @app.post("/predict/")
# async def predict_ecg(input_data: ECGInput):
#     if model is None:
#         return {"error": "Model not loaded correctly."}
    
#     try:
#         ecg_array = np.array(input_data.data).reshape(1, -1)  # Ensure correct shape
#         prediction = model.predict(ecg_array)  # Run the prediction
#         return {"prediction": int(prediction[0])}  # Convert to int for clarity
#     except Exception as e:
#         return {"error": str(e)}




# @app.post("/predict/")
# async def predict_ecg(input_data: ECGInput):
#     print("Received Data Length:", len(input_data.data))  # Debugging print

#     try:
#         # Resample input to exactly 187 features (if needed)
#         ecg_array = np.array(resample(input_data.data, 187)).reshape(1, -1)

#         # Make prediction
#         prediction = model.predict(ecg_array)

#         return {"prediction": int(prediction[0])}  # Convert to int for clarity
#     except Exception as e:
#         return {"error": str(e)}


@app.post("/predict/")
async def predict_ecg(input_data: ECGInput):
    print("Received Data Length:", len(input_data.data))  # Debugging print

    # Ensure input length matches the model's expected size
    if len(input_data.data) != 187:
        return {"error": f"Expected 187 features, but got {len(input_data.data)}"}

    try:
        ecg_array = np.array(input_data.data).reshape(1, -1)  # Ensure correct shape
        prediction = model.predict(ecg_array)  # Run the prediction
        return {"prediction": int(prediction[0])}  # Convert to int for clarity
    except Exception as e:
        return {"error": str(e)}
