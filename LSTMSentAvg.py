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
        sys.stdout = open('sentavg_GCDC_all_class_cos.txt', 'w')
        self.data_obj = data_obj
        self.task = params['task']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.lstm_dim = params['lstm_dim']
        self.dropout = params['dropout']
        self.embeddings = data_obj.word_embeds
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_dim)
        self.hidden = None
        self.hidden_layer = nn.Linear(50, self.hidden_dim)
        if params['task'] == 'perm':
            num_labels = 2
        elif params['task'] == 'minority':
            num_labels = 2
        elif params['task'] == 'class':
            num_labels = 3
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
        lstm_out = None  # document vectors
        global_deg_vec = []
        global_avg_deg =[]
        for i in range(len(inputs)):  # loop over docs
            doc_batch_size = len(inputs[i])  # number of sents
            self.hidden = self.init_hidden(doc_batch_size)
            seq_tensor = self.embeddings(inputs[i])
            # pack
            packed_input = pack_padded_sequence(
                seq_tensor, input_lengths[i], batch_first=True)
            # sentence representation
            packed_output, (ht, ct) = self.lstm(packed_input, self.hidden)
            # reorder
            final_output = ht[-1]
            odx = original_index[i].view(-1, 1).expand(
                len(input_lengths[i]), final_output.size(-1))
            output_unsorted = torch.gather(final_output, 0, Variable(odx))

            # 1. Faire une boucle sur le nombre de phrase dans un document, telle que dans chaque
            # itération on calcule le vecteur f entre 2 phrases adjacentes
            # print('========================= output sorted ====================')
            # print(output_unsorted)
            size = list(output_unsorted.size())
            print("================= SIZE ===============")
            print(size)

            # 2. calculer les degrès de continuité entre les fi du tensor précedent et construire un vecteur qui les regroupent.
            deg_vec = []

            if(size[0]==1):
                deg_vec.append(1)
            else:
                for i in range(size[0] - 1):
                    deg = nn.CosineSimilarity(dim=0, eps=1e-8)(output_unsorted[i], output_unsorted[i+1])
                    deg = deg.detach().numpy().item()
                    print("========================= deg  =======================")
                    print(deg)
                    # vecteur de degrés de continuité
                    deg_vec.append(deg)
            print("========================= deg vec =======================")
            print(deg_vec)
            # Padding the deg vec
            # calculate the avg of the deg vector of one document
            
            if(len(deg_vec) > 0):
                avg_deg= sum(deg_vec)/len(deg_vec)
                global_avg_deg.append(avg_deg)
            
            
            print("==================Global avg deg=====================")
            print(global_avg_deg)
                
            #global_avg_deg += deg_vec
            
            pad_deg = np.zeros(self.max_len)
            deg_vec = np.array(deg_vec)
            pad_deg[:deg_vec.size] = deg_vec
            print("==================Pad deg - initial tensor=====================")
            print(pad_deg)
            global_deg_vec.append(pad_deg)
            # append to get the avg deg vector of all documents 
            
        global_deg_vec = torch.FloatTensor(global_deg_vec)
        global_deg_vec = global_deg_vec.squeeze(1)
        print("==================Global deg vectors=====================")
        print(global_deg_vec)

        global_vectors = F.dropout(self.bn(F.relu(self.hidden_layer(
            global_deg_vec))), p=self.dropout, training=self.training)

        print("==================Global vectors=====================")
        print(global_vectors)

        coherence_pred = self.predict_layer(global_vectors)
        print("========================= COHERENCE PRED =======================")
        print(coherence_pred)

        if self.task != 'score_pred':
            coherence_pred = F.softmax(coherence_pred, dim=0)
        return coherence_pred, global_avg_deg
