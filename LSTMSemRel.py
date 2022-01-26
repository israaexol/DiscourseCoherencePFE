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

# todo this whole class
class LSTMSemRel(nn.Module):

    def __init__(self, params, data_obj):
        super(LSTMSemRel, self).__init__()
        sys.stdout = open('semrel_GCDC_class.txt', 'w')
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
        # self.par_lstm = nn.LSTM(self.lstm_dim, self.lstm_dim)
        # self.par_lstm_hidden = None
        self.hidden_layer = nn.Linear(50, self.hidden_dim)
        if params['task'] == 'perm':
            num_labels = 2
        elif params['task'] == 'minority':
            num_labels = 2
        elif params['task'] == 'class':
            num_labels = 1#3
        elif params['task'] == 'score_pred':
            num_labels = 1
        self.max_len = 50  # used for padding
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

    def forward(self, inputs, input_lengths, original_index):
        doc_vecs = None
        print("executing forward in lstmsemRel")
        #global cosine similarty vectors for sentences
        global_cosine_sent = [] # le degré de continuité de tous les documents d'un seul batch
        global_avg_cosine_sent = [] # les moyennes des cosine similarity des phrases d'un document
        
        #global cosine similarty vectors for paragraphs
        global_cosine_par = [] # le degré de continuité de tous les documents d'un seul batch
        global_avg_cosine_par = [] # les moyennes des cosine similarity des paragraphes d'un document
        for i in range(len(inputs)): # loop over docs
            par_vecs = None
            print("================Looping over docs===============")
            # storing global sentences from all paragraphs
            sentences_from_doc = torch.empty(0)
            for j in range(len(inputs[i])): # loop over paragraphs
                print("================Looping over parags===============")
                doc_batch_size = len(inputs[i][j]) # number of sents
                self.word_lstm_hidden = self.init_hidden(doc_batch_size)
                seq_tensor = self.embeddings(inputs[i][j])
                # pack
                packed_input = pack_padded_sequence(seq_tensor, input_lengths[i][j], batch_first=True)
                packed_output, (ht, ct) = self.word_lstm(packed_input, self.word_lstm_hidden)
                # reorder
                final_output = ht[-1] #a verifier
                odx = original_index[i][j].view(-1, 1).expand(len(input_lengths[i][j]), final_output.size(-1))

                 #1. get sentences per paragraph,

                output_unsorted = torch.gather(final_output, 0, Variable(odx))

                sentences_from_doc = torch.cat([sentences_from_doc, output_unsorted],dim=0)
                print("=========== Sentences from doc =================")
                print(type(sentences_from_doc))
                print(sentences_from_doc)
                # get paragraphs
                
                output_unsorted = output_unsorted.unsqueeze(1) #this is to get every sentence representation on its own in the tensor
                # LSTM to produce paragraph vector from sentence vectors
                self.sent_lstm_hidden = self.init_hidden(output_unsorted.size(1)) # batch size 1
                output_pars, (ht, ct) = self.sent_lstm(output_unsorted, self.sent_lstm_hidden)
                final_output = ht[-1]
                # append paragraph vector to batch
                if par_vecs is None:
                    par_vecs = final_output
                else:
                    par_vecs = torch.cat([par_vecs, final_output], dim=0)
            
            # SENTENCE LEVEL :
            # 2. calculate cosine similarities between sentences

            sent_size = list(sentences_from_doc.size())
            print("==================Sentences size ==================")
            print(sent_size)
            cosineSim_sent_doc = []

            if(sent_size[0]==1):
                cosineSim_sent_doc.append(1)
            else:
                for i in range(sent_size[0] - 1):
                    cosine_sent = nn.CosineSimilarity(dim=0, eps=1e-8)(sentences_from_doc[i], sentences_from_doc[i+1])
                    cosine_sent = cosine_sent.detach().numpy().item()
                    # vecteur de degrés de continuité
                    cosineSim_sent_doc.append(cosine_sent)

            print("=========== Cosine sent doc =================")
            print(type(cosineSim_sent_doc))
            print(cosineSim_sent_doc)
            #Retrieve all cosine similarities, padded
            if(len(cosineSim_sent_doc) > 0):
                avg_cosine= sum(cosineSim_sent_doc)/len(cosineSim_sent_doc)
                global_avg_cosine_sent.append(avg_cosine)
            pad_cosine = np.zeros(self.max_len)
            cosineSim_sent_doc = np.array(cosineSim_sent_doc)
            pad_cosine[:cosineSim_sent_doc.size] = cosineSim_sent_doc
            global_cosine_sent.append(pad_cosine)
            print("=========== Global cosine sent =================")
            print(type(global_cosine_sent))
            print(global_cosine_sent)
            ###############################################################################
                       
            # PARAGRAPH LEVEL :
            # 1. all paragraphs representations from one doc

            par_vecs = par_vecs.squeeze(1)
            par_size = par_vecs.size()           
            cosineSim_par_doc = [] #cosine similarity between all adjacent paragraphs per document

            # 2. calculate cosine similarities between paragraphs
            # make predictions
            
            if(par_size[0]==1):
                cosineSim_par_doc.append(1)
            else:
                for i in range(par_size[0]-1):
                    cosine_par = nn.CosineSimilarity(dim=0, eps=1e-8)(par_vecs[i], par_vecs[i+1])
                    cosine_par = cosine_par.detach().numpy().item()
                    # vecteur de degrés de continuité d'un seul document
                    cosineSim_par_doc.append(cosine_par)
            print("================Cosine sim par doc=============")
            print(cosineSim_par_doc)
            # avg of deg of continuity of one doc 
            if(len(cosineSim_par_doc) > 0): # vecteur de degrés de continuité d'un seul document
                avg_cosine = sum(cosineSim_par_doc)/len(cosineSim_par_doc) # avg of deg of continuity of one doc 
                global_avg_cosine_par.append(avg_cosine) # average of continuity degrees across all documents 
         
            #padding 

            pad_cosine_par = np.zeros(self.max_len)
            cosineSim_par_doc = np.array(cosineSim_par_doc)
            pad_cosine_par[:cosineSim_par_doc.size] = cosineSim_par_doc
            global_cosine_par.append(pad_cosine_par)
            print("=========== Global cosine par =================")
            print(type(global_cosine_par))
            print(global_cosine_par)
        # Prediction for Sentences

        global_cosine_sent = torch.FloatTensor(global_cosine_sent)
        global_cosine_sent  = global_cosine_sent.squeeze(1)
        global_vectors_sent = F.dropout(self.bn(F.relu(self.hidden_layer(global_cosine_sent))), p=self.dropout, training=self.training)
        coherence_pred_sent = self.predict_layer(global_vectors_sent)
        print("===========  Coherence pred sentences =================")
        print(type(coherence_pred_sent))
        print(coherence_pred_sent)

        ######################################################

        # Prediction for paragraphes 

        global_cosine_par = torch.FloatTensor(global_cosine_par)
        global_cosine_par = global_cosine_par.squeeze(1)
        global_vectors_par = F.dropout(self.bn(F.relu(self.hidden_layer(global_cosine_par))), p=self.dropout, training=self.training)
        coherence_pred_par = self.predict_layer(global_vectors_par)
        print("===========  Coherence pred paragraphs =================")
        print(type(coherence_pred_par))
        print(coherence_pred_par)
        # Linear layer for prediction weightage 
        coherence_pred_sent = coherence_pred_sent.flatten().tolist()
        coherence_pred_par = coherence_pred_par.flatten().tolist()
        print("===========  Coherence pred paragraphs =================")
        print(type(coherence_pred_par))
        print(coherence_pred_par)

        print("===========  Coherence pred sentences =================")
        print(type(coherence_pred_sent))
        print(coherence_pred_sent)
        coherence = {'sent' : coherence_pred_sent, 'par': coherence_pred_par}
        coh_dataframe = pd.DataFrame(coherence)
       
        
        print("===========  Coherence dataframe =================")
        print(type(coh_dataframe))
        print(coh_dataframe)
        
        #X = 
        # Final prediction 
        # if self.task != 'score_pred':
        #     final_coherence_pred = F.softmax(final_coherence_pred, dim=0)
        return coh_dataframe
           