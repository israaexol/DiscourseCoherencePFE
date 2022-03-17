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
        sys.stdout = open('CNNPosTag_CV_5.txt', 'w')
        self.data_obj = data_obj
        self.task = params['task']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.cnn_dim = params['lstm_dim']
        self.embeddings = data_obj.word_embeds

        #self.kernel_1 = 2
        self.kernel_1 = 3
        self.kernel_2 = 4
        self.kernel_3 = 5
        self.num_filters=[100, 100, 100]
        self.num_classes = 3

        # Number of strides for each convolution
        self.stride = 2

        # Convolution layers definition
        # self.conv_1 = nn.Conv2d(1, self.num_filters[0], (self.kernel_1, self.embedding_dim), padding=(self.kernel_1-2,0))
        # self.conv_2 = nn.Conv2d(1, self.num_filters[0], (self.kernel_2, self.embedding_dim), padding=(self.kernel_2-2,0))
        # self.conv_3 = nn.Conv2d(1, self.num_filters[0], (self.kernel_3, self.embedding_dim), padding=(self.kernel_3-2,0))

        self.conv_1 = nn.Conv1d(self.embedding_dim, self.cnn_dim, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.embedding_dim, self.cnn_dim, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.embedding_dim, self.cnn_dim, self.kernel_3, self.stride)

        # Max pooling layers definition
        # self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        # self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        # self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        # self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)
        
        # Fully connected layer definition
        self.linear = nn.Linear(np.sum(self.num_filters), self.num_classes)
        self.dropout = nn.Dropout(params['dropout'])
        #self.bn = nn.BatchNorm1d(300)

        #self.softmax = nn.Softmax()

        self.hidden = None
        self.max_len = 50  # used for padding
        # weight initialization
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

    def forward(self, inputs, input_lengths, original_index):
        global_coherence_pred = None
        for i in range(len(inputs)):  # loop over docs
            doc_batch_size = len(inputs[i])  # number of sents
            self.hidden = self.init_hidden(doc_batch_size)
            seq_tensor = self.embeddings(inputs[i])
            #seq_tensor = seq_tensor.unsqueeze(1)
            # print("======= seq_tensor size before permute =================")
            # print(seq_tensor.size())
            seq_tensor = seq_tensor.permute(0, 2, 1)
            #print("======= seq_tensor size =================")
            #print(seq_tensor.size())
            # Convolution layer 1 is applied
            #x1 = torch.relu(x1)
            #x1 = self.pool_1(x1)
            x1=F.relu(self.conv_1(seq_tensor))
            x1=F.max_pool1d(x1, kernel_size=x1.shape[2]).squeeze(2)
            # Convolution layer 2 is applied
            # x2 = self.conv_2(seq_tensor)
            # # x2 = torch.relu(x2)
            # # x2 = self.pool_2(x2)
            # x2=F.relu(x2)
            # x2=F.max_pool1d(x2, kernel_size=x2.shape[3])
            #x2 = x2.squeeze(dim=3)
            x2=F.relu(self.conv_1(seq_tensor))
            x2=F.max_pool1d(x2, kernel_size=x2.shape[2]).squeeze(2)
            # print("============ x2 size ==========")
            # print(x2.size())
            # Convolution layer 3 is applied
            x3=F.relu(self.conv_1(seq_tensor))
            x3=F.max_pool1d(x3, kernel_size=x3.shape[2]).squeeze(2)
            # print("============ x3 size ==========")
            # print(x3.size())
            
            # Convolution layer 4 is applied
            # x4=F.relu(self.conv_1(seq_tensor)).squeeze(3)
            # x4=F.max_pool1d(x4, kernel_size=x4.shape[2]).squeeze(2)
            # print("============ x4 size ==========")
            # print(x4.size())

            # The output of each convolutional layer is concatenated into a unique vector
            union = torch.cat((x1, x2, x3), 1)
            # print("=============== Union before reshape =============")
            # print(union)
            #print(union.size())
            # union = union.reshape(union.size(0), -1)
            # print("=============== Union after reshape =============")
            # print(union)
            # print(union.size())
           # The "flattened" vector is passed through a fully connected layer
            
            # Dropout is applied 2x300 300x3 2x3		
            out = self.dropout(union)
            # print("============== out dropout ================")
            # print(out)
            # print(out.size())
            out = self.linear(out)
            # print("============== out ================")
            # print(out)
            # print(out.size()) 
            
            if global_coherence_pred is None:
                global_coherence_pred = out
            else:
                global_coherence_pred = torch.cat([global_coherence_pred, out], dim=0)
        # print("============= Global coherence pred ==================")
        # print(global_coherence_pred)

        coherence_pred = F.softmax(global_coherence_pred, dim=0)
        # print("======= coherence pred from softmax ======")
        # print(coherence_pred)
        return coherence_pred
