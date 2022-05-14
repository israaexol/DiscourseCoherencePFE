from asyncio.log import logger
from lib2to3.pgen2 import token
import uvicorn
import pickle
import random
import numpy
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from LSTMSentAvg import LSTMSentAvg
from LSTMParSeq import LSTMParSeq
from LSTMSemRel import LSTMSemRel
from CNNPosTag import CNNPosTag
from DocumentWithParagraphs import DocumentWithParagraphs
from evaluation import eval_docs
from train_neural_models import train
from data_loader import *
import sys
import json
from json import JSONEncoder
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd
from tempfile import NamedTemporaryFile
import csv
from typing import Optional, List
from database import SessionLocal
from database import DATABASE_URL
from schema import Admin as SchemaAdmin
from schema import Model as SchemaModel
from schema import Inputs as Inputs

from schema import Token
from schema import TokenData
from models import Admin as ModelAdmin
from models import Model as ModelModel
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
app = FastAPI()
# db=SessionLocal()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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

sys.stdout = open('test.txt', 'w')


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)


# load the dictionary
embeddings = pickle.load(open('../model/word_embeds.pkl', 'rb'))
best_weights = pickle.load(open('../model/best_weights.pkl', 'rb'))
params = {
    'vector_type': 'glove'
}

dataObj = Data(params)
word_to_idx = pickle.load(open('word_to_idx.pkl', 'rb'))
idx_to_word = pickle.load(open('idx_to_word.pkl', 'rb'))
dataObj.word_embeds = embeddings
dataObj.word_to_idx = word_to_idx
dataObj.idx_to_word = idx_to_word
# if(dataObj.word_embeds == None):
#     vectors, vector_dim = dataObj.load_vectors()


def convert_csv(bytes):
    data = {}
    file_copy = NamedTemporaryFile(delete=False)
    try:
        with file_copy as f:
            f.write(bytes)

        with open(file_copy.name, 'r', encoding='utf-8') as csvf:
            csvReader = csv.DictReader(csvf)
            i = 0
            for rows in csvReader:
                key = i
                data[key] = rows
                i = i+1
    finally:
        file_copy.close()
        os.unlink(file_copy.name)
    return data


def generate_tags(text):
    stop = set(stopwords.words('english') + list(string.punctuation))
    text_tags = sent_tokenize(text)
    seq_tag = []
    for sentence in text_tags:
        sent_seq = ''
        b = []
        i = word_tokenize(sentence)
        for j in i:
            if j not in stop:
                b.append(j)
        i = b
        word_tag = nltk.pos_tag(i)

        for word in word_tag:
            sent_seq = sent_seq + word[1] + ' '
        seq_tag.append(sent_seq)
    print(seq_tag)
    tagged_text = ''
    length = len(seq_tag)
    i = 1
    for sent in seq_tag:
        if(i == length):
            tagged_text = tagged_text + sent
        else:
            tagged_text = tagged_text + sent
        i = i+1
    return tagged_text


def preprocess_data_sentavg(text):
    # read data class
    documents = []
    add_new_words = True
    text = text.lower()
    text_id = random.randint(0, 1000)
    label = None
    doc = DocumentWithParagraphs(text, label, id=text_id)
    doc_indexed = []
    for para in doc.text:
        para_indexed = []
        for sent in para:
            sent_indexed = []
            for word in sent:
                sent_indexed.append(
                    dataObj.add_token_to_index(word, add_new_words))
            para_indexed.append(sent_indexed)
        doc_indexed.append(para_indexed)
    doc.text_indexed = doc_indexed
    documents.append(doc)
    documents_data, documents_labels, documents_ids = dataObj.create_doc_sents(
        documents, 'sentence', 'class')
    indices = [int('0')]
    sentences, orig_batch_labels = dataObj.get_batch(
        documents_data, documents_labels, indices, 'sent_avg')
    batch_padded, batch_lengths, original_index = dataObj.pad_to_batch(
        sentences, dataObj.word_to_idx, 'sent_avg')
    return batch_padded, batch_lengths, original_index


def preprocess_data_parseq(text):
    # read data class
    documents = []
    add_new_words = True
    text = text.lower()
    text_id = random.randint(0, 1000)
    label = None
    doc = DocumentWithParagraphs(text, label, id=text_id)
    doc_indexed = []
    for para in doc.text:
        para_indexed = []
        for sent in para:
            sent_indexed = []
            for word in sent:
                sent_indexed.append(
                    dataObj.add_token_to_index(word, add_new_words))
            para_indexed.append(sent_indexed)
        doc_indexed.append(para_indexed)
    doc.text_indexed = doc_indexed
    documents.append(doc)
    documents_data, documents_labels, documents_ids = dataObj.create_doc_sents(
        documents, 'paragraph', 'class')
    indices = [int('0')]
    sentences, orig_batch_labels = dataObj.get_batch(
        documents_data, documents_labels, indices, 'par_seq')
    batch_padded, batch_lengths, original_index = dataObj.pad_to_batch(
        sentences, dataObj.word_to_idx, 'par_seq')
    return batch_padded, batch_lengths, original_index


def preprocess_data_semrel(text):
    # read data class
    documents = []
    add_new_words = True
    text = text.lower()
    text_id = random.randint(0, 1000)
    label = None
    doc = DocumentWithParagraphs(text, label, id=text_id)
    doc_indexed = []
    for para in doc.text:
        para_indexed = []
        for sent in para:
            sent_indexed = []
            for word in sent:
                sent_indexed.append(
                    dataObj.add_token_to_index(word, add_new_words))
            para_indexed.append(sent_indexed)
        doc_indexed.append(para_indexed)
    doc.text_indexed = doc_indexed
    documents.append(doc)
    documents_data, documents_labels, documents_ids = dataObj.create_doc_sents(
        documents, 'paragraph', 'class')
    indices = [int('0')]
    sentences, orig_batch_labels = dataObj.get_batch(
        documents_data, documents_labels, indices, 'sem_rel')
    batch_padded, batch_lengths, original_index = dataObj.pad_to_batch(
        sentences, dataObj.word_to_idx, 'sem_rel')
    return batch_padded, batch_lengths, original_index


def preprocess_data_cnnpostag(text):
    # read data class
    documents = []
    add_new_words = True
    text = text.lower()
    text = generate_tags(text)
    text_id = random.randint(0, 1000)
    label = None
    doc = DocumentWithParagraphs(text, label, id=text_id)
    doc_indexed = []
    for para in doc.text:
        para_indexed = []
        for sent in para:
            sent_indexed = []
            for word in sent:
                sent_indexed.append(
                    dataObj.add_token_to_index(word, add_new_words))
            para_indexed.append(sent_indexed)
        doc_indexed.append(para_indexed)
    doc.text_indexed = doc_indexed
    documents.append(doc)
    documents_data, documents_labels, documents_ids = dataObj.create_doc_sents(
        documents, 'paragraph', 'class')
    indices = [int('0')]
    sentences, orig_batch_labels = dataObj.get_batch(
        documents_data, documents_labels, indices, 'cnn_pos_tag')
    batch_padded, batch_lengths, original_index = dataObj.pad_to_batch(
        sentences, dataObj.word_to_idx, 'cnn_pos_tag')
    return batch_padded, batch_lengths, original_index
# Setting up the home route


@app.get("/")
def read_root():
    return {"data": "Welcome to La Coherencia"}


@app.post("/evaluate")
async def get_predict(data: Inputs):
    niveau = data.selectedIndex
    sample = data.text

    if niveau == 0:
        #model = LSTMSentAvg(params, data_obj=dataObj)
        #model = model.load_state_dict(torch.load('../model/runs/sentavg_model/sentavg_model_best'))
        #model = pickle.load(open('../model/sent_avg.pkl', 'rb'))
        model = torch.load(
            '../model/runs/sent_avg_model/sent_avg_model_best.pt')
        model.eval()
        batch_padded, batch_lengths, original_index = preprocess_data_sentavg(
            sample)
        print('===================batch_padded===================')
        print(batch_padded)
        pred, avg_deg = model.forward(
            batch_padded, batch_lengths, original_index, dim=1)
        print('====================pred=========================')
        print(pred)
        argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)

    elif niveau == 1:
        model = torch.load('../model/runs/par_seq_model/par_seq_model_best.pt')
        model.eval()
        batch_padded, batch_lengths, original_index = preprocess_data_parseq(
            sample)
        pred, avg_deg = model.forward(
            batch_padded, batch_lengths, original_index, dim=1)
        argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
    elif niveau == 2:
        model = torch.load('../model/runs/semrel_model/semrel_model_best.pt')
        model.eval()
        batch_padded, batch_lengths, original_index = preprocess_data_semrel(
            sample)
        pred = model.forward(batch_padded, batch_lengths,
                             original_index, weights=best_weights, dim=1)
        argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
    elif niveau == 3:
        model = torch.load(
            '../model/runs/cnn_postag_model/cnn_postag_model_best.pt')
        model.eval()
        batch_padded, batch_lengths, original_index = preprocess_data_cnnpostag(
            sample)
        pred = model.forward(batch_padded, batch_lengths,
                             original_index, dim=1)
        argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
    else:
        model = pickle.load(open('../model/sem_syn_cv.pkl', 'rb'))

    return {
        "data": {
            'score': score
            # 'interpretation': 'Candidate can be hired.' if label == 1 else 'Candidate can not be hired.'
        }
    }


@app.post("/uploadfile")
async def get_predict_file(niveau: int, file: UploadFile = File(...)):
    content_assignment = await file.read()
    data = convert_csv(content_assignment)
    print("============data==============")
    print(data)
    scores = []
    text_ids = []
    texts = []
    original_scores = []
    for i in range(len(data)):
        sample = data[i]['text']
        text_ids.append(data[i]['text_id'])
        texts.append(data[i]['text'])
        original_scores.append(data[i]['labelA'])
        if niveau == 0:
            model = torch.load(
                '../model/runs/sent_avg_model/sent_avg_model_best.pt')
            model.eval()
            batch_padded, batch_lengths, original_index = preprocess_data_sentavg(
                sample)
            # print('===================batch_padded===================')
            # print(batch_padded)
            pred, avg_deg = model.forward(
                batch_padded, batch_lengths, original_index, dim=1)
            # print('====================pred=========================')
            # print(pred)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)

        elif niveau == 1:
            model = torch.load(
                '../model/runs/par_seq_model/par_seq_model_best.pt')
            model.eval()
            batch_padded, batch_lengths, original_index = preprocess_data_parseq(
                sample)
            pred, avg_deg = model.forward(
                batch_padded, batch_lengths, original_index, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
        elif niveau == 2:
            model = torch.load(
                '../model/runs/semrel_model/semrel_model_best.pt')
            model.eval()
            batch_padded, batch_lengths, original_index = preprocess_data_semrel(
                sample)
            pred = model.forward(batch_padded, batch_lengths,
                                 original_index, weights=best_weights, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
        elif niveau == 3:
            model = torch.load(
                '../model/runs/cnn_postag_model/cnn_postag_model_best.pt')
            model.eval()
            batch_padded, batch_lengths, original_index = preprocess_data_cnnpostag(
                sample)
            pred = model.forward(
                batch_padded, batch_lengths, original_index, dim=1)
            argmax = list(np.argmax(pred.cpu().data.numpy(), axis=1))
            score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
            scores.append(score)
        else:
            model = pickle.load(open('../model/sem_syn_cv.pkl', 'rb'))

    print(scores)
    return {"data":  {"scores": scores, "text_ids": text_ids, "texts": texts, "original_scores": original_scores}}

# Retourner les liste de tous les modèles avec les détails pour l'interface admin


@app.get('/models/', response_model=List[SchemaModel], status_code=200)
def get_all_models(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    list_models = db.query(ModelModel).all()
    return list_models

# Retourner les détails d'un modèle spécifié par son Id pour l'interface de modification côté admin


@app.get('/models/{model_id}', response_model=SchemaModel, status_code=200)
def get_one_model(model_id: int, db: Session = Depends(get_db)):
    db_model = db.query(ModelModel).filter(
        ModelModel.id == model_id).first()
    if db_model is None:
        raise HTTPException(status_code=404, detail="Model non existant")
    return db_model

# Retourner la description d'un modèle précis pour sidebar


@app.get('/description/{model_id}', status_code=200)
def get_one_model(model_id: int, db: Session = Depends(get_db)):
    db_model = db.query(ModelModel.description).filter(
        ModelModel.id == model_id and ModelModel.visibility == True).first()
    if db_model is None:
        raise HTTPException(status_code=404, detail="Model non existant")
    return db_model

# Retourner les noms des modèles à afficher dans la liste déroulante des modèles existants


@app.get('/names/', status_code=200)
def get_models_name(db: Session = Depends(get_db)):
    db_model = db.query(ModelModel.id, ModelModel.name).filter(
        ModelModel.visibility == True).first()
    if db_model is None:
        raise HTTPException(status_code=404, detail="Aucun modèle n'éxiste")
    return db_model

# Ajouter un modèle dans l'interface d'ajout côté admin


@app.post('/add_model', response_model=SchemaModel)
def add_model(model: SchemaModel, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    db_model_name = db.query(ModelModel).filter(
        ModelModel.name == model.name).first()
    if db_model_name:
        raise HTTPException(status_code=400, detail="Modèle déjà existant")
    else:
        db_model = ModelModel(name=model.name, description=model.description,
                              F1_score=model.F1_score, precision=model.precision, accuracy=model.accuracy)
        db.add(db_model)
        db.commit()
        return db_model
# Modifier les paramètres d'un modèle


@app.put("/update_model/{model_id}", response_model=SchemaModel)
def update_model(model_id: int, model: SchemaModel, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    model_to_update = db.query(ModelModel).filter(
        ModelModel.id == model_id).first()
    if model.name:
        model_to_update.name = model.name
    if model.description:
        model_to_update.description = model.description
    if model.F1_score:
        model_to_update.F1_score = model.F1_score
    if model.precision:
        model_to_update.precision = model.precision
    if model.accuracy:
        model_to_update.accuracy = model.accuracy
    db.commit()
    return model_to_update

# Modifier la visibilité d'un modèle


@app.put("/update_model_visibility/{model_id}", response_model=SchemaModel)
def update_model_vis(model_id: int, visib: bool, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    model_to_update = db.query(ModelModel).filter(
        ModelModel.id == model_id).first()
    model_to_update.visibility = visib
    db.commit()
    return model_to_update
# Login


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_admin(username: str, db: Session = Depends(get_db)):
    result = db.query(ModelAdmin).filter(
        ModelAdmin.user_name == username).first()
    return result


def authenticate_user(username: str, password: str, db: Session = Depends(get_db)):
    admin = get_admin(username, db)
    if not admin:
        return False
    if not verify_password(password, admin.pwd):
        return False
    return admin


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Login


@app.post("/login", response_model=Token)
async def login_for_access_token(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Nom d'utilisateur ou mot de passe est incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.user_name}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


def get_password_hash(password):
    return pwd_context.hash(password)

# To add a new admin


@app.post('/sign_up', response_model=SchemaAdmin)
def add_admin(admin: SchemaAdmin, db: Session = Depends(get_db)):
    db_admin_name = db.query(ModelModel).filter(
        ModelAdmin.user_name == admin.user_name).first()
    if db_admin_name:
        raise HTTPException(status_code=400, detail="Modèle déjà existant")
    else:
        db_admin = ModelAdmin(user_name=admin.user_name,
                              pwd=get_password_hash(admin.pwd))
        db.add(db_admin)
        db.commit()
        return db_admin


# Configuring the server host and port
if __name__ == '__main__':
    uvicorn.run("main:app", port=8080, host='0.0.0.0', reload=True)
