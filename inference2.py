
import os
import warnings
import numpy as np
# Suppress TensorFlow logs and oneDNN custom operations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import StandardScaler
# Suppress all warnings
warnings.filterwarnings('ignore')
# Further suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
import uvicorn
import pytz
from typing import Union, Optional, List
from fastapi import FastAPI, Depends, status, Request ,HTTPException
from fastapi.security import HTTPBasic ,HTTPBasicCredentials
from pydantic import BaseModel , ValidationError
from pydantic import Field ,validator
from enum import Enum
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File, HTTPException
import io
from pathlib import Path
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from tensorflow.keras.preprocessing import image  
from pydantic import BaseModel, Field
from typing import Optional
import pickle 


# Load your saved model
model = tf.keras.models.load_model(r'C:\Users\DELL\Desktop\DRD\vgg16_with_additional_features_multiclass.h5')
save_directory = Path(r"C:\Users\DELL\Desktop\patient_data")
app1 = FastAPI(title="Diabetic Retinopathy Detection using AI")
# Ensure the directory exists
save_directory.mkdir(parents=True, exist_ok=True)

scaler_file=r"C:\Users\DELL\Desktop\DRD\additional_features_scaler.pkl"
def load_scalar(scaler_file):
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)
        return scaler
# Function to predict class for a single record
async def process_and_predict(image_file,file_path,additional_features,scaler_file):
    contents = await image_file.read()
    # Save the image to the specified file_path
    with open(file_path, 'wb') as f:
        f.write(contents)
    image_data = io.BytesIO(contents)
    img = image.load_img(image_data, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

  
    # Convert additional features to numpy array and scale
    additional_features = np.array(additional_features).reshape(1, -1)
    scaler=load_scalar(scaler_file)
    additional_features = scaler.transform(additional_features)

   
    predictions = model.predict([img_array, additional_features])
    print(predictions)
    # Get predicted class and confidence
    predicted_class = np.argmax(predictions[0])
    print(predicted_class)
    confidence = round(predictions[0][predicted_class]*100)
    class_values={0:"No Retinopathy",
                  1:"Mild Stage",
                  2:"Moderate",
                  3:"Severe",
                  4:"Proliferative DR"}
    stage=class_values[predicted_class]
    print(stage)
    #Map the predicted class to its explanation
    explanation_labels = {
        0: 'No DR: No diabetic retinopathy detected.',
        1: 'Mild: Early stage of diabetic retinopathy with small areas of balloon-like swelling in the retina’s tiny blood vessels.',
        2: 'Moderate: More severe than mild with more extensive areas of blood vessel blockage in the retina.',
        3: 'Severe: Significant blood vessel blockage leading to deprivation of blood supply to several areas of the retina.',
        4: 'Proliferative DR: Most advanced stage with growth of new blood vessels, which can lead to serious vision problems.'
    }
    explanation = explanation_labels[predicted_class]
    Risk = (100 - confidence) 
    


    result= {
        "predicted_class": int(predicted_class),
        "confidence": float(confidence),
        "explanation": explanation,
        "warning": None,
        "Risk_Factor":Risk,
        "Stage":stage
        }
    
    if confidence < 55 and predicted_class >= 0 and predicted_class <=3:
        result["warning"] = f"You have a higher chance of progressing to the next stage with risk factor {Risk}%. Please consult your doctor for further advice."
    elif confidence > 55 and predicted_class >= 0 and predicted_class <=3:
        result["warning"] = f"You have a little chance of progressing to the next stage with risk factor {Risk}%"
    elif confidence >= 75 and predicted_class == 0:
        result["warning"] = "Your eye is in the safe zone."
    elif confidence > 55 and confidence <=74 and predicted_class == 0:
        result["warning"]=f"you have minimum chance of risk for prgressing to next level you risk factor {Risk}%"
    elif  predicted_class == 4:
        result["warning"] = "Alert: Significant signs of diabetic retinopathy detected. Your eye is at a high risk, and immediate medical attention is crucial to prevent potential vision loss. Please consult your healthcare provider urgently."


    # Return the structured result
    return {
        "predicted_class": result["predicted_class"],
        "stage":result["Stage"],
        "confidence": result["confidence"],
        "explanation": result["explanation"],
        "Note": result["warning"],
        "Risk" :f"{Risk}%",
        }


class AdditionalFeatures(BaseModel):
    HbA1c: Optional[float] = Field(0.0, description="Hemoglobin A1c level")
    Systolic_BP: Optional[float] = Field(0.0, description="Systolic blood pressure")
    Diastolic_BP: Optional[float] = Field(0.0, description="Diastolic blood pressure")
    LDL: Optional[float] = Field(0.0, description="Low-density lipoprotein")
    Duration: Optional[float] = Field(0.0, description="Duration of diabetes in years")
    BMI: Optional[float] = Field(0.0, description="Body mass index")
    Glucose_SD: Optional[float] = Field(0.0, description="Standard deviation of glucose levels")
    Triglycerides: Optional[float] = Field(0.0, description="Triglyceride levels")
    Microalbuminuria: Optional[float] = Field(0.0, description="Microalbuminuria level")
    Smoking_years: Optional[int] = Field(0, description="Number of years smoking")
    Alcohol_frequency: Optional[int] = Field(0, description="Frequency of alcohol consumption")
    BP: Optional[float] = Field(0.0, description="Blood pressure")

def insert_into_db(
    patient_id, eye_type,Prediction, Stage, Confidence, RISK, Explanation, NOTE, Ingested_date,Image_path,HbA1c, Systolic_BP, 
    Diastolic_BP, LDL, Duration, BMI, Glucose_SD, Triglycerides, Microalbuminuria, Smoking_years, Alcohol_frequency, BP):
    try:
        # Establish the connection
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password='iscs_user',
            database="patient"
        )
        
        if connection.is_connected():
           
            cursor = connection.cursor()
            insert_query=""" INSERT INTO patient_data(patient_id, eye_type, Prediction, Stage, Confidence, RISK, Explanation, 
            NOTE, Ingested_date, Image_path, HbA1c, Systolic_BP, Diastolic_BP, LDL, Duration, BMI, Glucose_SD, Triglycerides, Microalbuminuria, 
            Smoking_years, Alcohol_frequency, BP
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""


            record_to_insert = (
                patient_id,eye_type,Prediction,Stage,Confidence,RISK,Explanation,NOTE,Ingested_date,Image_path
                ,HbA1c, Systolic_BP, Diastolic_BP, LDL, Duration, BMI, Glucose_SD, Triglycerides, 
                                                    Microalbuminuria, 
                                                    Smoking_years, 
                                                    Alcohol_frequency, 
                                                    BP
            )
            cursor.execute(insert_query, record_to_insert)
            connection.commit()
            print("Data inserted successfully")
        else:
            print("Failed to connect to database")
    except Error as e:
        print("Error while inserting data into MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
# Get predictions and confidences
@app1.post("/Prediction")
async def predict_image_class(
    patient_id: int,
    right_eye: UploadFile = File(...), 
    left_eye: UploadFile = File(...),
    features: AdditionalFeatures = Depends()):
    additional_features = [
        features.HbA1c, features.Systolic_BP, features.Diastolic_BP, features.LDL, 
        features.Duration, features.BMI, features.Glucose_SD, features.Triglycerides, 
        features.Microalbuminuria, features.Smoking_years, features.Alcohol_frequency, features.BP
    ]
    
    print(additional_features)
    """Predict the class of diabetic retinopathy for right and left eye images."""

    # Construct the full file paths for right and left eye images
    right_eye_path = save_directory / f"{patient_id}_right_eye.jpg"
    left_eye_path = save_directory / f"{patient_id}_left_eye.jpg"

    # Process and predict for the right eye
    right_eye_result = await process_and_predict(right_eye, right_eye_path,additional_features,scaler_file)
    print(right_eye_result)
     # Extract details from the result
    predicted_class = right_eye_result["predicted_class"]
    stage=right_eye_result["stage"]
    explanation = right_eye_result["explanation"]
    confidence = right_eye_result["confidence"]
    risk = right_eye_result["Risk"]
    eye_side="Right Eye"
    Note=right_eye_result["Note"]
    ist = pytz.timezone('Asia/Kolkata')  # Indian Standard Time
    ist_time = datetime.now(ist)
    # Convert IST time to MySQL datetime format (YYYY-MM-DD HH:MM:SS)
    ist_time_str = ist_time.strftime('%Y-%m-%d %H:%M:%S')
      
    insert_into_db(
        patient_id,              # INT: Unique identifier for the patient
        eye_side,                # VARCHAR(50): Type of eye or side, e.g., 'Left' or 'Right'
        predicted_class,         # INT: Predicted condition or stage
        stage, 
        confidence, 
        risk,              # VARCHAR(50): Stage of condition
        explanation,             # TEXT: Explanation text for the predictio           # INT: Confidence percentage   
        Note,                    # TEXT: Additional notes
        ist_time_str, 
        str(right_eye_path),                 # TEXT: Date and time when the data was ingested
        features.HbA1c,          # FLOAT: Hemoglobin A1c level
        features.Systolic_BP,    # FLOAT: Systolic blood pressure
        features.Diastolic_BP,   # FLOAT: Diastolic blood pressure
        features.LDL,            # FLOAT: Low-density lipoprotein
        features.Duration,       # FLOAT: Duration of diabetes in years
        features.BMI,            # FLOAT: Body mass index
        features.Glucose_SD,     # FLOAT: Standard deviation of glucose levels
        features.Triglycerides,  # FLOAT: Triglyceride levels
        features.Microalbuminuria, # FLOAT: Microalbuminuria level
        features.Smoking_years,  # INT: Number of years smoking
        features.Alcohol_frequency, # INT: Frequency of alcohol consumption
        features.BP              # FLOAT: Blood pressure
    )

    # Process and predict for the left eye
    left_eye_result = await process_and_predict(left_eye, left_eye_path,additional_features,scaler_file)
    predicted_class = left_eye_result["predicted_class"]
    stage=left_eye_result["stage"]
    explanation = left_eye_result["explanation"]
    confidence = left_eye_result["confidence"]
    risk = left_eye_result["Risk"]
    eye_side="Left Eye"
    Note=left_eye_result["Note"]
    insert_into_db(
        patient_id,              # INT: Unique identifier for the patient
         # TEXT: Path to the image (converted to string)
        eye_side,                # VARCHAR(50): Type of eye or side, e.g., 'Left' or 'Right'
        predicted_class,         # INT: Predicted condition or stage
        stage, 
        confidence, 
        risk,              # VARCHAR(50): Stage of condition
        explanation,             # TEXT: Explanation text for the predictio           # INT: Confidence percentage   
        Note,                    # TEXT: Additional notes
        ist_time_str, 
        str(left_eye_path) ,         # TEXT: Date and time when the data was ingested
        features.HbA1c,          # FLOAT: Hemoglobin A1c level
        features.Systolic_BP,    # FLOAT: Systolic blood pressure
        features.Diastolic_BP,   # FLOAT: Diastolic blood pressure
        features.LDL,            # FLOAT: Low-density lipoprotein
        features.Duration,       # FLOAT: Duration of diabetes in years
        features.BMI,            # FLOAT: Body mass index
        features.Glucose_SD,     # FLOAT: Standard deviation of glucose levels
        features.Triglycerides,  # FLOAT: Triglyceride levels
        features.Microalbuminuria, # FLOAT: Microalbuminuria level
        features.Smoking_years,  # INT: Number of years smoking
        features.Alcohol_frequency, # INT: Frequency of alcohol consumption
        features.BP              # FLOAT: Blood pressure
    )

    

  

    return JSONResponse(content={
        "patient_id": patient_id,
        "right_eye": right_eye_result,
        "left_eye": left_eye_result
    })
@app1.get("/Getting the patient records")
def get_patient_data():
    try:
        # Establish the connection
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password='iscs_user',
            database="patient"
        )
        
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)  # Use dictionary=True to return results as dicts
            cursor.execute("SELECT * FROM patient_data")
            
            # Fetch the results
            results = cursor.fetchall()
            
            if not results:
                raise HTTPException(status_code=404, detail="No patient records found")
                
            return results
    
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Error while connecting to MySQL: {str(e)}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app1, host="0.0.0.0", port=8000)