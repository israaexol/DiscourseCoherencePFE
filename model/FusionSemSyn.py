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
class FusionSemSyn(nn.Module):

    def __init__(self, params, data_cnn, data_semrel, data_obj):
        super(FusionSemSyn, self).__init__()
        sys.stdout = open('fusion_GCD.txt', 'w')
        
        ###########################################################
        #common attributes
        self.params = params
        self.data_cnn = data_cnn
        self.data_semrel = data_semrel
        self.task = params['task']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.dropout = params['dropout']
        self.embeddings_cnn = data_obj.word_embeds
        self.embeddings_semrel = data_obj.word_embeds
        
        ###########################################################
        #SemRel related attributes
        self.lstm_dim = params['lstm_dim']
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
        
        ###########################################################
        #CNN Pos Tag related attributes
        self.cnn_dim = params['lstm_dim']
        self.kernel_1 = 1
        self.kernel_2 = 2
        self.kernel_3 = 3
        self.kernel_4 = 4
        #self.kernel_5 = 5
        self.num_filters=[100, 100, 100, 100]
        self.num_classes = 3
        # Number of strides for each convolution
        self.stride = 2
        
        # Dans le cas d'utilisation d'un ensemble filtré de tags
        if params['tag_filter'] == 1:
            self.conv_1 = nn.Conv2d(1, 100, (self.kernel_1, self.embedding_dim), padding=(1,0))    
        else : 
            self.conv_1 = nn.Conv1d(self.embedding_dim, self.cnn_dim, self.kernel_1, self.stride)
            self.conv_2 = nn.Conv1d(self.embedding_dim, self.cnn_dim, self.kernel_2, self.stride)
            self.conv_3 = nn.Conv1d(self.embedding_dim, self.cnn_dim, self.kernel_3, self.stride)
            self.conv_4 = nn.Conv1d(self.embedding_dim, self.cnn_dim, self.kernel_4, self.stride)
            #self.conv_5 = nn.Conv2d(1, 100, (self.kernel_5, self.embedding_dim), padding=(3,0))
            
        self.linearCNN = nn.Linear(np.sum(self.num_filters), self.num_classes)
        self.dropoutCNN = nn.Dropout(params['dropout'])
        self.hiddenCNN = None

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

    def forward(self, inputsSemRel, inputsCNN, input_lengthsSemRel, input_lengthsCNN, original_index, weights=None, dim=0):
        
        ###########################################################
        ###########################################################
        # Partie de traitement relatif à SemRel
        doc_vecs = None
        global_cosine_sent = [] # les similarités cosines de tous les documents d'un seul batch
        global_avg_cosine_sent = [] # les moyennes des similarités cosines des phrases d'un document
        
        global_cosine_par = [] # les similarités cosines de tous les documents d'un seul batch
        global_avg_cosine_par = [] # les moyennes des similarités cosines des paragraphes d'un document
        for i in range(len(inputsSemRel)): # itérer sur les documents
            par_vecs = None
            # Utilisée pour récupérer les phrases par document
            sentences_from_doc = torch.empty(0)
            for j in range(len(inputsSemRel[i])): # itérer sur les paragraphes
                doc_batch_size = len(inputsSemRel[i][j]) # nombre de phrases
                self.word_lstm_hidden = self.init_hidden(doc_batch_size)
                seq_tensor = self.embeddings_semrel(inputsSemRel[i][j])
                # pack
                packed_input = pack_padded_sequence(seq_tensor, input_lengthsSemRel[i][j], batch_first=True)                
                # génération des représentations de phrases à partir des word embeddings
                packed_output, (ht, ct) = self.word_lstm(packed_input, self.word_lstm_hidden)
                # réordonner
                final_output = ht[-1]
                odx = original_index[i][j].view(-1, 1).expand(len(input_lengthsSemRel[i][j]), final_output.size(-1))

                output_unsorted = torch.gather(final_output, 0, Variable(odx))

                sentences_from_doc = torch.cat([sentences_from_doc, output_unsorted],dim=0)
                
                output_unsorted = output_unsorted.unsqueeze(1)
                self.sent_lstm_hidden = self.init_hidden(output_unsorted.size(1))
                # génération des représentations de paragraphes à partir de celles des phrases
                output_pars, (ht, ct) = self.sent_lstm(output_unsorted, self.sent_lstm_hidden)
                final_output = ht[-1]
                
                # concaténer les vecteurs de paragraphes
                if par_vecs is None:
                    par_vecs = final_output
                else:
                    par_vecs = torch.cat([par_vecs, final_output], dim=0)
            
            
            ####################### SIMILARITÉS COSINES NIVEAU PHRASES ##############################
            sent_size = list(sentences_from_doc.size())
            cosineSim_sent_doc = []

            if(sent_size[0]==1):
                cosineSim_sent_doc.append(1)
            else:
                for i in range(sent_size[0] - 1):
                    cosine_sent = nn.CosineSimilarity(dim=0, eps=1e-8)(sentences_from_doc[i], sentences_from_doc[i+1])
                    cosine_sent = cosine_sent.detach().numpy().item()
                    # vecteur de similarités cosine
                    cosineSim_sent_doc.append(cosine_sent)

            if(len(cosineSim_sent_doc) > 0):
                avg_cosine= sum(cosineSim_sent_doc)/len(cosineSim_sent_doc)
                global_avg_cosine_sent.append(avg_cosine)
            
            # padding et concaténation
            pad_cosine = np.zeros(self.max_len)
            cosineSim_sent_doc = np.array(cosineSim_sent_doc)
            pad_cosine[:cosineSim_sent_doc.size] = cosineSim_sent_doc
            global_cosine_sent.append(pad_cosine)
            
            ####################### COSINE SIMILARITY NIVEAU PARAGRAPHES ##############################
            # représentations des paragraphes par document

            par_vecs = par_vecs.squeeze(1)
            par_size = par_vecs.size()           
            cosineSim_par_doc = [] #similarités cosine entre les paragraphes adjacents d'un document

            if(par_size[0]==1):
                cosineSim_par_doc.append(1)
            else:
                for i in range(par_size[0]-1):
                    cosine_par = nn.CosineSimilarity(dim=0, eps=1e-8)(par_vecs[i], par_vecs[i+1])
                    cosine_par = cosine_par.detach().numpy().item()
                    # vecteur de degrés de continuité d'un seul document
                    cosineSim_par_doc.append(cosine_par)

            if(len(cosineSim_par_doc) > 0): # vecteur de similarités cosines d'un seul document
                avg_cosine = sum(cosineSim_par_doc)/len(cosineSim_par_doc) # moyenne des similarités cosine d'un seul document 
                global_avg_cosine_par.append(avg_cosine) # concaténer pour avoir les moyennes des similarités cosines de tous les documents
         
            # padding et concaténation
            pad_cosine_par = np.zeros(self.max_len)
            cosineSim_par_doc = np.array(cosineSim_par_doc)
            pad_cosine_par[:cosineSim_par_doc.size] = cosineSim_par_doc
            global_cosine_par.append(pad_cosine_par)
            
        #Prédiction niveau phrases
        global_cosine_sent = torch.FloatTensor(global_cosine_sent)
        global_cosine_sent  = global_cosine_sent.squeeze(1)
        global_vectors_sent = F.dropout(self.bn(F.relu(self.hidden_layer(global_cosine_sent))), p=self.dropout, training=self.training)
        coherence_pred_sent = self.predict_layer(global_vectors_sent)
        
        # Tâche de classification
        coherence_pred_sent = F.softmax(coherence_pred_sent, dim=0)
        coherence_pred_sentTensor = coherence_pred_sent
    
        ######################################################
        # Prédiction niveau paragraphe 

        global_cosine_par = torch.FloatTensor(global_cosine_par)
        global_cosine_par = global_cosine_par.squeeze(1)
        global_vectors_par = F.dropout(self.bn(F.relu(self.hidden_layer(global_cosine_par))), p=self.dropout, training=self.training)
        coherence_pred_par = self.predict_layer(global_vectors_par)

        coherence_pred_par = F.softmax(coherence_pred_par, dim=0)
        coherence_pred_parTensor = coherence_pred_par
        
        ###########################################################
        ###########################################################
        # Partie de traitement relatif à CNN_POS_Tag
        global_coherence_pred = None
        for i in range(len(inputsCNN)):  # Itérer sur les documents
            doc_batch_size = len(inputsCNN[i])  # nombre de phrases
            self.hiddenCNN = self.init_hidden(doc_batch_size)
            embeds = self.embeddings_cnn(inputsCNN[i])
            #seq_tensor = seq_tensor.unsqueeze(1)
            
            #if conv1D
            embeds = embeds.permute(0, 2, 1)

            if self.params['tag_filter'] == 1:
                x1=F.relu(self.conv_1(embeds)).squeeze(3)
                x1=F.max_pool1d(x1, kernel_size=x1.shape[2]).squeeze(2)
                union = x1
            else: 
                x1=F.relu(self.conv_1(embeds))
                x1=F.max_pool1d(x1, kernel_size=x1.shape[2]).squeeze(2)
                
                # Appliquer Convolution layer 2
                x2=F.relu(self.conv_2(embeds))
                x2=F.max_pool1d(x2, kernel_size=x2.shape[2]).squeeze(2)
                
                # Appliquer Convolution layer 3
                x3=F.relu(self.conv_3(embeds))
                x3=F.max_pool1d(x3, kernel_size=x3.shape[2]).squeeze(2)
            
                # Appliquer Convolution layer 4
                x4=F.relu(self.conv_4(embeds))
                x4=F.max_pool1d(x4, kernel_size=x4.shape[2]).squeeze(2)
            
                # Appliquer Convolution layer 5
                #x5=F.relu(self.conv_5(seq_tensor))
                #x5=F.max_pool1d(x5, kernel_size=x5.shape[2]).squeeze(2)

                # La sortie de chaque couche de convolution est concaténée dans un unique tensor
                union = torch.cat((x1, x2, x3, x4), 1)

            # Le vecteur est fourni au fully connected layer
            # Appliquer le Dropout	
            out = self.dropoutCNN(union)
            out = self.linearCNN(out)
            
            if global_coherence_pred is None:
                global_coherence_pred = out
            else:
                # concaténer les prédictions issues des différents documents du batch
                global_coherence_pred = torch.cat([global_coherence_pred, out], dim=0)

        # Pour la classification
        coherence_pred_CNN = F.softmax(global_coherence_pred, dim=dim)
        coherence_pred_CNNTensor = coherence_pred_CNN
        
        #Prédiction finale à partir de celles des deux niveaux
        return coherence_pred_sent, coherence_pred_par, coherence_pred_CNN
           