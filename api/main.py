import uvicorn
import pickle
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]
app.add_middleware(
CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Inputs(BaseModel):
    text: str
    niveau : int

@app.post("/evaluate/")
async def get_predict(data: Inputs):
    id = 1
    sample = [[
        id,
        data.text
    ]]
    if data.niveau == 1:
        model = pickle.load(open('../model/sent_avg_cv.pkl', 'rb'))
    elif data.niveau == 2:
        model = pickle.load(open('../model/par_seq_cv.pkl', 'rb'))
    elif data.niveau == 3:
        model = pickle.load(open('../model/sem_rel_cv.pkl', 'rb'))
    elif data.niveau == 4:
        model = pickle.load(open('../model/cnn_pos_tag_cv.pkl', 'rb'))
    else : 
        model = pickle.load(open('../model/sem_syn_cv.pkl', 'rb'))

    label = model.predict(data.text)
    return {
    "data": {
    'score': label
    #'interpretation': 'Candidate can be hired.' if label == 1 else 'Candidate can not be hired.'
}
}

model = pickle.load(open('../model/hireable.pkl', 'rb'))