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


class CNNPosTag(nn.Module):

    def __init__(self, params, data_obj):
        super(CNNPosTag, self).__init__()
        self.params = params
        sys.stdout = open('CNN-Yahoo.txt', 'w')
        self.data_obj = data_obj
        self.task = params['task']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.cnn_dim = params['lstm_dim']
        self.embeddings = data_obj.word_embeds

        self.kernel_1 = 1
        self.kernel_2 = 2
        self.kernel_3 = 3
        self.kernel_4 = 4
        #self.kernel_5 = 5
        self.num_filters=[100, 100, 100, 100]
        if params['task'] == 'class':
            self.num_classes = 3
        elif params['task'] == 'score_pred':
            self.num_classes = 1

        # Number of strides for each convolution
        self.stride = 2

        # self.conv_1 = nn.Conv2d(1, 100, (self.kernel_1, self.embedding_dim), padding=(1,0))    
        # self.conv_2 = nn.Conv2d(1, 100, (self.kernel_2, self.embedding_dim), padding=(0,0))
        # self.conv_3 = nn.Conv2d(1, 100, (self.kernel_3, self.embedding_dim), padding=(1,0))
        # self.conv_4 = nn.Conv2d(1, 100, (self.kernel_4, self.embedding_dim), padding=(2,0))
        
        #self.conv_5 = nn.Conv1d(self.embedding_dim, self.cnn_dim, self.kernel_5, self.stride)

        # Dans le cas d'utilisation d'un ensemble filtré de tags
        if params['tag_filter'] == 1:
            self.conv_1 = nn.Conv2d(1, 100, (self.kernel_1, self.embedding_dim), padding=(1,0))    
        else : 
            self.conv_1 = nn.Conv1d(self.embedding_dim, self.cnn_dim, self.kernel_1, self.stride)
            self.conv_2 = nn.Conv1d(self.embedding_dim, self.cnn_dim, self.kernel_2, self.stride)
            self.conv_3 = nn.Conv1d(self.embedding_dim, self.cnn_dim, self.kernel_3, self.stride)
            self.conv_4 = nn.Conv1d(self.embedding_dim, self.cnn_dim, self.kernel_4, self.stride)
            #self.conv_5 = nn.Conv2d(1, 100, (self.kernel_5, self.embedding_dim), padding=(3,0))
        
        # La définition d'une couche Fully connected
        self.linear = nn.Linear(np.sum(self.num_filters), self.num_classes)
        self.dropout = nn.Dropout(params['dropout'])
        self.hidden = None
        
        # initialisation des poids
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
                nn.init.xavier_uniform(m.weight)
        if USE_CUDA:
            self.hidden_layer = self.hidden_layer.cuda()
            self.linear = self.linear.cuda()

    def init_hidden(self, batch_size):
        if USE_CUDA:
            return (Variable(torch.zeros(1, batch_size, self.cnn_dim).cuda()),
                    Variable(torch.zeros(1, batch_size, self.cnn_dim)).cuda())
        else:
            return (Variable(torch.zeros(1, batch_size, self.cnn_dim)),
                    Variable(torch.zeros(1, batch_size, self.cnn_dim)))

    def forward(self, inputs, input_lengths, original_index, dim=0):
        global_coherence_pred = None
        for i in range(len(inputs)):  # Itérer sur les documents
            doc_batch_size = len(inputs[i])  # nombre de phrases
            self.hidden = self.init_hidden(doc_batch_size)
            seq_tensor = self.embeddings(inputs[i])
            #seq_tensor = seq_tensor.unsqueeze(1)
            
            #if conv1D
            seq_tensor = seq_tensor.permute(0, 2, 1)

            if self.params['tag_filter'] == 1:
                x1=F.relu(self.conv_1(seq_tensor)).squeeze(3)
                x1=F.max_pool1d(x1, kernel_size=x1.shape[2]).squeeze(2)
                union = x1
            else: 
                x1=F.relu(self.conv_1(seq_tensor))
                x1=F.max_pool1d(x1, kernel_size=x1.shape[2]).squeeze(2)
                
                # Appliquer Convolution layer 2
                x2=F.relu(self.conv_2(seq_tensor))
                x2=F.max_pool1d(x2, kernel_size=x2.shape[2]).squeeze(2)
                
                # Appliquer Convolution layer 3
                x3=F.relu(self.conv_3(seq_tensor))
                x3=F.max_pool1d(x3, kernel_size=x3.shape[2]).squeeze(2)
            
                # Appliquer Convolution layer 4
                x4=F.relu(self.conv_4(seq_tensor))
                x4=F.max_pool1d(x4, kernel_size=x4.shape[2]).squeeze(2)
            
                # Appliquer Convolution layer 5
                #x5=F.relu(self.conv_5(seq_tensor))
                #x5=F.max_pool1d(x5, kernel_size=x5.shape[2]).squeeze(2)

                # La sortie de chaque couche de convolution est concaténée dans un unique tensor
                union = torch.cat((x1, x2, x3, x4), 1)

            # Le vecteur est fourni au fully connected layer
            # Appliquer le Dropout	
            out = self.dropout(union)
            out = self.linear(out)
            
            if global_coherence_pred is None:
                global_coherence_pred = out
            else:
                # concaténer les prédictions issues des différents documents du batch
                global_coherence_pred = torch.cat([global_coherence_pred, out], dim=0)

        coherence_pred = global_coherence_pred
        # Pour la classification
        if self.task != 'score_pred':
            coherence_pred = F.softmax(global_coherence_pred, dim=dim) # classification des documents
        
        return coherence_pred
