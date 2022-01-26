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

# todo this whole class
class LSTMParSeq(nn.Module):

    def __init__(self, params, data_obj):
        super(LSTMParSeq, self).__init__()
        sys.stdout = open('parseq_GCDC_Clinton_class.txt', 'w')
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
        doc_vecs = None
        global_deg_par = [] # le degré de continuité de tous les documents d'un seul batch
        global_avg_deg_doc = [] # les moyennes des degrés de continuité des paragraphes d'un document
        for i in range(len(inputs)): # loop over docs
            par_vecs = None
            for j in range(len(inputs[i])): # loop over paragraphs
                doc_batch_size = len(inputs[i][j]) # number of sents
                self.word_lstm_hidden = self.init_hidden(doc_batch_size)
                seq_tensor = self.embeddings(inputs[i][j])
                # pack
                packed_input = pack_padded_sequence(seq_tensor, input_lengths[i][j], batch_first=True)
                packed_output, (ht, ct) = self.word_lstm(packed_input, self.word_lstm_hidden)
                # reorder
                final_output = ht[-1] #a verifier
                odx = original_index[i][j].view(-1, 1).expand(len(input_lengths[i][j]), final_output.size(-1))
                output_unsorted = torch.gather(final_output, 0, Variable(odx))
                # LSTM to produce paragraph vector from sentence vectors
                output_unsorted = output_unsorted.unsqueeze(1) #this is to get every sentence representation on its own in the tensor
                self.sent_lstm_hidden = self.init_hidden(output_unsorted.size(1)) # batch size 1
                output_pars, (ht, ct) = self.sent_lstm(output_unsorted, self.sent_lstm_hidden)
                final_output = ht[-1]
                # append paragraph vector to batch
                if par_vecs is None:
                    par_vecs = final_output
                else:
                    par_vecs = torch.cat([par_vecs, final_output], dim=0)
        
            par_vecs = par_vecs.squeeze(1)
            
            #par_vecs = par_vecs.unsqueeze(1)
            print("=====================Paragraphe vectors=====================")
            print(par_vecs)
            print("=====================Paragraphe vectors size=====================")
            print(par_vecs.size())
            size = par_vecs.size()           
            deg_par_vec = [] #cosine similarity between all adjacent paragraphs per document
            # Cosine similarity between paragraphes of one document
            if(size[0]==1):
                deg_par_vec.append(1)
            else:
                for i in range(size[0]-1):
                    deg_par = nn.CosineSimilarity(dim=0, eps=1e-8)(par_vecs[i], par_vecs[i+1])
                    deg_par = deg_par.detach().numpy().item()
                    print("========================= DEG PAR =======================")
                    print(deg_par)
                    # vecteur de degrés de continuité d'un seul document
                    deg_par_vec.append(deg_par)
            
            print("========================= DEG PAR VEC =======================")
            print(deg_par_vec)

            # avg of deg of continuity of one doc 
            if(len(deg_par_vec) > 0): # vecteur de degrés de continuité d'un seul document
                avg_deg_doc= sum(deg_par_vec)/len(deg_par_vec) # avg of deg of continuity of one doc 
                global_avg_deg_doc.append(avg_deg_doc) # average of continuity degrees across all documents 
            print("===========================Global avg deg doc=====================")
            print(global_avg_deg_doc)

            #padding 

            pad_deg = np.zeros(self.max_len)
            deg_par_vec = np.array(deg_par_vec)
            pad_deg[:deg_par_vec.size] = deg_par_vec
            print("==================Pad deg - initial tensor=====================")
            print(pad_deg)
            global_deg_par.append(pad_deg)
            print("=====================Global deg par==============================")
            print(global_deg_par) # deg de continuité des documents d'un seul batch après padding
            print("=======================global deg par Length=====================")
            print(len(global_deg_par))
            
        global_deg_par = torch.FloatTensor(global_deg_par)
        global_deg_par = global_deg_par.squeeze(1)
        print("==================Global deg vectors=====================")
        print(global_deg_par)
          
        global_vectors = F.dropout(self.bn(F.relu(self.hidden_layer(global_deg_par))), p=self.dropout, training=self.training)
        print("==================Global vectors=====================")
        print(global_vectors)
        coherence_pred = self.predict_layer(global_vectors)
        print("==========================Coherence prediction =============================")
        print(coherence_pred)
        if self.task != 'score_pred':
            coherence_pred = F.softmax(coherence_pred, dim=0)
        return coherence_pred, global_avg_deg_doc
