import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import sys


USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


class LSTMSentAvg(nn.Module):

    def __init__(self, params, data_obj):
        super(LSTMSentAvg, self).__init__()
        sys.stdout = open('sentavg_logs.txt', 'w')
        self.data_obj = data_obj
        self.task = params['task']
        self.embedding_dim = params['embedding_dim']
        self.hidden_dim = params['hidden_dim']
        self.lstm_dim = params['lstm_dim']
        self.dropout = params['dropout']
        self.embeddings = data_obj.word_embeds
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_dim)
        self.hidden = None
        self.hidden_layer = nn.Linear(self.lstm_dim, self.hidden_dim)
        if params['task'] == 'perm':
            num_labels = 2
        elif params['task'] == 'minority':
            num_labels = 2
        elif params['task'] == 'class':
            num_labels = 3
        elif params['task'] == 'score_pred':
            num_labels = 1
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
        for i in range(len(inputs)):  # loop over docs
            doc_batch_size = len(inputs[i])  # number of sents
            self.hidden = self.init_hidden(doc_batch_size)
            seq_tensor = self.embeddings(inputs[i])
            # pack
            packed_input = pack_padded_sequence(seq_tensor, input_lengths[i], batch_first=True)
            packed_output, (ht, ct) = self.lstm(packed_input, self.hidden)
            # reorder
            final_output = ht[-1]
            #FIRST LSTM OUTPUT
            # tensor([[[-0.0030, -0.0869, -0.1645,  0.0495, -0.0060,  0.0708,  0.2143,
            #           -0.0974, -0.0781, -0.0135,  0.0143,  0.1013, -0.0372, -0.1258,
            #           -0.0904, -0.0533,  0.1858, -0.1266, -0.1215,  0.0398,  0.0635,
            #           -0.0478, -0.1165, -0.1011,  0.1136, -0.0591, -0.1412, -0.0770,
            #            0.0721, -0.0613, -0.1329,  0.0216,  0.0608,  0.0317,  0.0228,
            #           -0.1503,  0.0578,  0.0431,  0.0107,  0.0236, -0.1482, -0.0477,
            #           -0.2412,  0.1409,  0.0077,  0.0582, -0.0100,  0.0666, -0.0217,
            #           -0.2402, -0.0395, -0.0039,  0.0029,  0.0411,  0.1129, -0.0430,
            #            0.0506, -0.1113, -0.0773,  0.0347,  0.1174, -0.0290, -0.0879,
            #           -0.1226, -0.0004, -0.1638,  0.0684,  0.0200, -0.0007,  0.0674,
            #           -0.1275,  0.0935,  0.0231,  0.0585,  0.1514, -0.0809,  0.0876,
            #           -0.0261,  0.1511, -0.0555, -0.0586,  0.0966, -0.0588, -0.0604,
            #            0.0722,  0.0870,  0.0672, -0.0873, -0.0778,  0.0207, -0.0654,
            #           -0.1568,  0.0028,  0.0654,  0.0660, -0.2314,  0.0937,  0.0095,
            #            0.0488, -0.0363]],

            #         [[-0.0612, -0.0574, -0.2097, -0.0240, -0.0117,  0.0837,  0.1836,
            #           -0.1650, -0.0279, -0.0866,  0.0314, -0.0071, -0.0703, -0.1794,
            #           -0.1765,  0.0554,  0.1147, -0.0096, -0.0843,  0.1257, -0.0468,
            #           -0.0706, -0.0245, -0.1029,  0.1240, -0.0570, -0.1717, -0.1275,
            #           -0.0008, -0.0630, -0.0102,  0.0444,  0.0499,  0.0528,  0.0124,
            #           -0.0387,  0.0510,  0.0476,  0.0930,  0.0229, -0.0375, -0.0369,
            #           -0.1694,  0.1720, -0.0504,  0.0557,  0.0052,  0.0168,  0.0737,
            #           -0.1589,  0.0347, -0.0149, -0.0580,  0.0537,  0.0422, -0.1184,
            #            0.2054, -0.0876, -0.0675,  0.0115,  0.0569, -0.1120, -0.0825,
            #           -0.1208, -0.1129, -0.2607,  0.1005, -0.1117, -0.0243, -0.0368,
            #           -0.1214,  0.1078, -0.0064,  0.0611,  0.2789, -0.1182, -0.0092,
            #           -0.0358,  0.2525, -0.0377, -0.1099,  0.0393, -0.0759, -0.0614,
            #           -0.0057,  0.0317, -0.0497, -0.1302, -0.0934,  0.0289, -0.1507,
            #           -0.1649,  0.0504,  0.0233, -0.0417, -0.2384,  0.1597,  0.0510,
            #            0.0102,  0.0325]]])
            odx = original_index[i].view(-1, 1).expand(len(input_lengths[i]), final_output.size(-1))
            output_unsorted = torch.gather(final_output, 0, Variable(odx))
            # sum sentence vectors
            # output_sum = torch.sum(output_unsorted, 0).unsqueeze(0)
            # if lstm_out is None:
            #     lstm_out = output_sum
            # else:
            #     lstm_out = torch.cat([lstm_out, output_sum], dim=0)
            
            #1. Faire une boucle sur le nombre de phrase dans un document, telle que dans chaque itération on calcule le vecteur f entre 2 phrases adjacentes
            print('========================= output sorted ====================')
            print(output_unsorted)
            size= list(output_unsorted.size())
            F_vectors=[]
            for i in range(size[0] - 2):
                 f = torch.stack([output_unsorted[i,0:size[1]], output_unsorted[i+1,0:size[1]]], dim=1)
                 f = torch.mean(f, dim=1)
            #2. à la sortie de la boucle, obtenir un tensor contenant tous les vecteurs f du document
                 F_vectors.append(f)
            #Print the F_vectors 
            print("========================= F_vectors =======================")
            print(F_vectors)
           
            #3. calculer les dergrès de continuité entre les f du tensor précedent et construire un vecteur qui les regroupent.
            deg_vec=[]
            for i in range(size[0] - 3):
               # f_trans = F_vectors[i+1].transpose(2,1) # .t()
            
                deg = torch.matmul(F_vectors[i],F_vectors[i+1])
                # prevent big values
                deg = torch.div(deg, size[1]) 
                deg = torch.nn.LeakyReLU(-1.0)(deg) 
                #vecteur de degrés de continuité
                deg_vec.append(deg)
            print("========================= deg vec =======================")
            print(deg_vec)
        #doc_vectors = F.dropout(self.bn(F.relu(self.hidden_layer(lstm_out))), p=self.dropout, training=self.training)
        deg_vectors = F.dropout(self.bn(F.relu(self.hidden_layer(deg_vec))), p=self.dropout, training=self.training)
        print("========================= deg_vectors =======================")
        print(deg_vectors)
        coherence_pred = self.predict_layer(deg_vectors)
        print("========================= COHERENCE PRED =======================")
        print(coherence_pred)
        if self.task != 'score_pred':
            coherence_pred = F.softmax(coherence_pred, dim=0)
        return coherence_pred
