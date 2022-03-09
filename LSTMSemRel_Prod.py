import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import sys
import numpy as np
import pandas as pd
from sklearn import linear_model


USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

# Une deuxième architecture pour la fusion des deux niveaux (phrases et paragraphes) pour un calcul global des cosine simialrités
# et étude de leur effet sur l'évaluation de la cohérence des documents

class LSTMSemRel_Prod(nn.Module):

    def __init__(self, params, data_obj):
        super(LSTMSemRel_Prod, self).__init__()
        sys.stdout = open('semrelprod_GCDC_10cv.txt', 'w')
        self.data_obj = data_obj
        self.task = params['task']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.lstm_dim = params['lstm_dim']
        self.dropout = params['dropout']
        self.embeddings = data_obj.word_embeds
        self.word_lstm = nn.LSTM(self.embedding_dim, self.lstm_dim)
        self.word_lstm_hidden = None
        self.sent_lstm = nn.LSTM(self.lstm_dim, self.lstm_dim)
        self.sent_lstm_hidden = None
        self.hidden_layer = nn.Linear(50, self.hidden_dim)
        if params['task'] == 'perm':
            num_labels = 2
        elif params['task'] == 'minority':
            num_labels = 2
        elif params['task'] == 'class':
            num_labels = 3
        elif params['task'] == 'score_pred':
            num_labels = 1
        self.max_len = 50  # maximum size pour padding
        self.predict_layer = nn.Linear(self.hidden_dim, num_labels)
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.xavier_uniform(m.weight)
        if USE_CUDA:
            self.hidden_layer = self.hidden_layer.cuda()
            self.predict_layer = self.predict_layer.cuda()

    def init_hidden(self, batch_size):
        if USE_CUDA:
            return (Variable(torch.zeros(1, batch_size, self.lstm_dim).cuda()),
                    Variable(torch.zeros(1, batch_size, self.lstm_dim)).cuda())
        else:
            return (Variable(torch.zeros(1, batch_size, self.lstm_dim)),
                    Variable(torch.zeros(1, batch_size, self.lstm_dim)))

    def forward(self, inputs, input_lengths, original_index, weights=None):
        doc_vecs = None
        #vecteurs des cosines similarités après fusion des deux niveaux
        global_cosine = []
     
        for i in range(len(inputs)): # itérer sur les documents
            par_vecs = None
            cosineSim_doc=[]
            global_cosine_sent = [] # le degré de continuité de tous les documents d'un seul batch
           
            
            for j in range(len(inputs[i])): #  itérer sur les paragraphes
                doc_batch_size = len(inputs[i][j]) # nombre de phrases
                self.word_lstm_hidden = self.init_hidden(doc_batch_size)
                seq_tensor = self.embeddings(inputs[i][j])
                # pack
                packed_input = pack_padded_sequence(seq_tensor, input_lengths[i][j], batch_first=True)
                # génération des représentations de phrases à partir des word embeddings
                packed_output, (ht, ct) = self.word_lstm(packed_input, self.word_lstm_hidden)
                # reorder
                final_output = ht[-1] #a verifier
                odx = original_index[i][j].view(-1, 1).expand(len(input_lengths[i][j]), final_output.size(-1))

                output_unsorted = torch.gather(final_output, 0, Variable(odx))
                
                ####################### COSINE SIMILARITY NIVEAU PHRASES ##############################
                sent_size = list(output_unsorted.size())
                cosineSim_sent_par = []
                
                if(sent_size[0]==1):
                    cosineSim_sent_par.append(1) 
                else:
                    for t in range(sent_size[0] - 1): 
                        cosine_sent = nn.CosineSimilarity(dim=0, eps=1e-8)(output_unsorted[t], output_unsorted[t+1])
                        cosine_sent = cosine_sent.detach().numpy().item()
                        # vecteur de degrés de continuité
                        cosineSim_sent_par.append(cosine_sent)
                global_cosine_sent.append(cosineSim_sent_par) # regrouper les cosine similarités des phrases (par paragraphe par document)

                output_unsorted = output_unsorted.unsqueeze(1)
                # génération des représentations de paragraphes à partir de celles des phrases
                self.sent_lstm_hidden = self.init_hidden(output_unsorted.size(1))
                output_pars, (ht, ct) = self.sent_lstm(output_unsorted, self.sent_lstm_hidden)
                final_output = ht[-1]
                # concaténer les vecteurs de paragraphes
                if par_vecs is None:
                    par_vecs = final_output
                else:
                    par_vecs = torch.cat([par_vecs, final_output], dim=0)
            
                ####################### COSINE SIMILARITY NIVEAU PARAGRAPHES ##############################
            # représentations des paragraphes par document
            par_vecs = par_vecs.squeeze(1)
            par_size = par_vecs.size()        
            cosineSim_par_doc = [] #cosine similarity entre les paragraphes adjacents d'un document
           
            if(par_size[0]==1):
                cosineSim_par_doc.append(1)
            else:
                for t in range(par_size[0]-1):
                    cosine_par = nn.CosineSimilarity(dim=0, eps=1e-8)(par_vecs[t], par_vecs[t+1])
                    cosine_par = cosine_par.detach().numpy().item()
                    # vecteur de cosine similarity d'un seul document
                    cosineSim_par_doc.append(cosine_par)
        
            for t in range(len(global_cosine_sent)):
                if(t==0):
                    cosineSim_doc += global_cosine_sent[t]
                else: 
                    for k in range(len(global_cosine_sent[t])):
                        cosineSim = global_cosine_sent[t][k] * cosineSim_par_doc[t-1]
                        cosineSim_doc.append(cosineSim)
       
            #padding 
            pad_cosine_doc = np.zeros(self.max_len)
            cosineSim_doc = np.array(cosineSim_doc)
            pad_cosine_doc[:cosineSim_doc.size] = cosineSim_doc
            global_cosine.append(pad_cosine_doc)

        global_cosine = torch.FloatTensor(global_cosine)
      
        global_cosine  = global_cosine.squeeze(1)
        global_vectors = F.dropout(self.bn(F.relu(self.hidden_layer(global_cosine))), p=self.dropout, training=self.training) #dropout
        coherence_pred = self.predict_layer(global_vectors)

        coherence_pred = F.softmax(coherence_pred, dim=0) #prédiction


        return coherence_pred
           