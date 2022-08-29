# Load the libraries
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn # pip install uvicorn[standard]
from joblib import load

# Load the model
gnb = load(open('./models/test_model.pkl','rb'))


# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Sentiment Classification FastAPI"}

class Item(BaseModel):
    SepalLengthCm: float = Field(gt=4.3, le=7.9, description="a must be greater equal than zero and less equal than 1")
    SepalWidthCm: float = Field(gt=2, le=4.4, description="b must be greater equal than zero and less equal than 1")
    PetalLengthCm: float = Field(gt=1, le=6.9, description="c must be greater equal than zero and less equal than 1")
    PetalWidthCm: float = Field(gt=0.1, le=2.5, description="d must be greater equal than zero and less equal than 1")


# Define the route to the test
@app.post("/predict_iris")
def predict_iris(item: Item):

    input_arr = np.array([[
        item.SepalLengthCm, 
        item.SepalWidthCm, 
        item.PetalLengthCm, 
        item.PetalWidthCm]])
        
    pred = gnb.predict(input_arr)[0]

    flower=""
    if pred==0:
        flower="Setosa"
    elif pred==1:
        flower="Versicolor"
    elif pred==2:
        flower="Virginica"
    
    return {
            "text_message": item, 
            "flower": flower
           }


if __name__ == "__main__":

    # REMARK
    # If you need to debug then launch the app.py from play button in VS
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # Otherwise if you want to start the API application on command line
    # to update always the application thanks to "reload" (you don't need to comment previous line) then:
    # uvicorn app:app --reload

    # DOCKER COMMAND
    # docker build -t fastapiapp:latest -f docker/Dockerfile .
    # docker run -p 80:80 fastapiapp:latest