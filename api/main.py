import uvicorn
import pickle
import random
import numpy
from pydantic import BaseModel
from fastapi import FastAPI
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
    
class Inputs(BaseModel):
    text: str
    
# load the dictionary
#embeddings = pickle.load(open('../model/word_embeds.pkl', 'rb'))
best_weights = pickle.load(open('../model/best_weights.pkl', 'rb'))
params = {
    'vector_type': 'glove'
}
dataObj = Data(params)
if(dataObj.word_embeds == None):   
    vectors, vector_dim = dataObj.load_vectors()
#dataObj.word_embeds = embeddings

def generate_tags(text):
    stop = set(stopwords.words('english')+ list(string.punctuation))
    text_tags = sent_tokenize(text) 
    seq_tag=[]
    for sentence in text_tags:
        sent_seq = ''
        b = []
        i= word_tokenize(sentence)
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
    documents_data, documents_labels, documents_ids = dataObj.create_doc_sents(documents, 'sentence', 'class')
    indices = [int('0')]
    sentences, orig_batch_labels = dataObj.get_batch(documents_data, documents_labels, indices, 'sent_avg')
    batch_padded, batch_lengths, original_index = dataObj.pad_to_batch(sentences, dataObj.word_to_idx, 'sent_avg')
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
    documents_data, documents_labels, documents_ids = dataObj.create_doc_sents(documents, 'paragraph', 'class')
    indices = [int('0')]
    sentences, orig_batch_labels = dataObj.get_batch(documents_data, documents_labels, indices, 'par_seq')
    batch_padded, batch_lengths, original_index = dataObj.pad_to_batch(sentences, dataObj.word_to_idx, 'par_seq')
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
    documents_data, documents_labels, documents_ids = dataObj.create_doc_sents(documents, 'paragraph', 'class')
    indices = [int('0')]
    sentences, orig_batch_labels = dataObj.get_batch(documents_data, documents_labels, indices, 'sem_rel')
    batch_padded, batch_lengths, original_index = dataObj.pad_to_batch(sentences, dataObj.word_to_idx, 'sem_rel')
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
    documents_data, documents_labels, documents_ids = dataObj.create_doc_sents(documents, 'paragraph', 'class')
    indices = [int('0')]
    sentences, orig_batch_labels = dataObj.get_batch(documents_data, documents_labels, indices, 'cnn_pos_tag')
    batch_padded, batch_lengths, original_index = dataObj.pad_to_batch(sentences, dataObj.word_to_idx, 'cnn_pos_tag')
    return batch_padded, batch_lengths, original_index
# Setting up the home route

@app.get("/")
def read_root():
    return {"data": "Welcome to La Coherencia"}


@app.post("/evaluate")
async def get_predict(data: Inputs):
    niveau = 2
    sample = data.text
  
    if niveau == 1:
        #model = LSTMSentAvg(params, data_obj=dataObj)
        #model = model.load_state_dict(torch.load('../model/runs/sentavg_model/sentavg_model_best'))
        #model = pickle.load(open('../model/sent_avg.pkl', 'rb'))
        model = torch.load('../model/runs/sent_avg_model/sent_avg_model_best.pt')
        model.eval()
        batch_padded, batch_lengths, original_index = preprocess_data_sentavg(sample)
        print('===================batch_padded===================')
        print(batch_padded)
        pred, avg_deg = model.forward(batch_padded, batch_lengths, original_index, dataObj.word_embeds, dim = 1)
        print('====================pred=========================')
        print(pred)
        argmax  = list(np.argmax(pred.cpu().data.numpy(), axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
      
    elif niveau == 2:
        model = torch.load('../model/runs/par_seq_model/par_seq_model_best.pt')
        model.eval()
        batch_padded, batch_lengths, original_index = preprocess_data_parseq(sample)
        print('===================batch_padded===================')
        print(batch_padded)
        pred, avg_deg = model.forward(batch_padded, batch_lengths, original_index, dataObj.word_embeds, dim = 1)
        print('====================pred=========================')
        print(pred)
        argmax  = list(np.argmax(pred.cpu().data.numpy(), axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
    elif niveau == 3:
        model = torch.load('../model/runs/semrel_model/semrel_model_best')
        model.eval()
        batch_padded, batch_lengths, original_index = preprocess_data_semrel(sample)
        print('===================batch_padded===================')
        print(batch_padded)
        pred = model.forward(batch_padded, batch_lengths, original_index, dataObj.word_embeds, weights=best_weights, dim = 1)
        print('====================pred=========================')
        print(pred)
        argmax  = list(np.argmax(pred.cpu().data.numpy(), axis=1))
        score = json.dumps(argmax[0], cls=NumpyArrayEncoder)
    elif niveau == 4:
        model = torch.load('../model/runs/cnn_postag_model/cnn_postag_model_best.pt')
        model.eval()
        batch_padded, batch_lengths, original_index = preprocess_data_cnnpostag(sample)
        print('===================batch_padded===================')
        print(batch_padded)
        pred = model.forward(batch_padded, batch_lengths, original_index, dataObj.word_embeds, dim = 1)
        print('====================pred=========================')
        print(pred)
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
    

# Configuring the server host and port
if __name__ == '__main__':
    uvicorn.run("main:app", port=8080, host='0.0.0.0', reload=True)
