import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import sys
import numpy as np

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


class LSTMSentAvg(nn.Module):

    def __init__(self, params, data_obj):
        super(LSTMSentAvg, self).__init__()
        sys.stdout = open('sentavg-clinton.txt', 'w')
        self.data_obj = data_obj
        self.task = params['task']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.lstm_dim = params['lstm_dim']
        self.dropout = params['dropout'] 
        self.embeddings = data_obj.word_embeds # couche pour générer les word embeddings
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_dim) # couche LSTM pour générer les representation vectorielle des phrases
        self.hidden = None
        self.hidden_layer = nn.Linear(50, self.hidden_dim) 

        if params['task'] == 'class':
            num_labels = 3
        elif params['task'] == 'score_pred':
            num_labels = 1

        self.max_len = 50  # used for padding
        self.predict_layer = nn.Linear(self.hidden_dim, num_labels) # couche de prédiction du score de cohérence des documents du batch
        self.bn = nn.BatchNorm1d(self.hidden_dim)

        # initialisation des poids
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

    def forward(self, inputs, input_lengths, original_index, dim = 0):
        global_deg_vec = []
        global_avg_deg =[]
        for i in range(len(inputs)):  # itérer sur les documents
            doc_batch_size = len(inputs[i])  # nombre de phrases
            self.hidden = self.init_hidden(doc_batch_size)
            seq_tensor = self.embeddings(inputs[i])
            # pack
            packed_input = pack_padded_sequence(
                seq_tensor, input_lengths[i], batch_first=True)
            # représentation des phrases
            packed_output, (ht, ct) = self.lstm(packed_input, self.hidden)
            # réordonner
            final_output = ht[-1]
            odx = original_index[i].view(-1, 1).expand(
                len(input_lengths[i]), final_output.size(-1))
            output_unsorted = torch.gather(final_output, 0, Variable(odx))

            # boucler sur le nombre de phrase dans un document, telle que dans chaque
            # itération on calcule le vecteur de similarité cosine (f) entre 2 phrases adjacentes
            size = list(output_unsorted.size())

            deg_vec = [] # vecteur des similarités cosine entre les phrases d'un seul document

            if(size[0]==1):
                deg_vec.append(1)
            else:
                for i in range(size[0] - 1):
                    deg = nn.CosineSimilarity(dim=0, eps=1e-8)(output_unsorted[i], output_unsorted[i+1])
                    deg = deg.detach().numpy().item()
                    # vecteur des similarités cosine
                    deg_vec.append(deg)
                    
            # calculer la moyenne des similarités cosine d'un document (utilisé pour la reprentation graphique)
            
            if(len(deg_vec) > 0):
                avg_deg= sum(deg_vec)/len(deg_vec)
                #Concaténation des moyennes de similarités cosine de tous les documents
                global_avg_deg.append(avg_deg)

            # Padding du vecteur de similarité cosine
            pad_deg = np.zeros(self.max_len)
            deg_vec = np.array(deg_vec)
            pad_deg[:deg_vec.size] = deg_vec
            global_deg_vec.append(pad_deg)
            
        global_deg_vec = torch.FloatTensor(global_deg_vec)
        global_deg_vec = global_deg_vec.squeeze(1)

        global_vectors = F.dropout(self.bn(F.relu(self.hidden_layer(
            global_deg_vec))), p=self.dropout, training=self.training)
        # La prédiction de cohérence pour chaque document du batch
        coherence_pred = self.predict_layer(global_vectors)

        if self.task != 'score_pred':
            coherence_pred = F.softmax(coherence_pred, dim=dim) #prédiction
        return coherence_pred #global_avg_deg
