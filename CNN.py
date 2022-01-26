
      # CNN layers
        self.conv = nn.Conv1d(in_channels=1,
                  out_channels=1,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=True)
            
        # max pooling
        self.max_pooling = nn.MaxPool2d(kernel_size = (1,3), stride = 1)
        self.dropout_layer = nn.Dropout(0.5)
        self.linear_5 = nn.Linear(self.max_len-4, 100, bias = True)
        self.tanh = nn.Tanh()



            # CNN layer === 133

           # pad_deg = torch.cat(pad_deg,dim=1)
            # print("==================Pad deg - torch.cat=====================")
            # print(pad_deg)

            pad_deg= self.dropout_layer(pad_deg)
            print("=======================Pad deg - droput_layer=================")
            print(pad_deg)
            # learn patterns in coh_output 
            pad_deg = pad_deg.unsqueeze(1)
            print("===================== Pad deg unsequeeze=================")
            print(pad_deg)

            pad_deg = self.conv(pad_deg)
            print("===================== Pad deg conv================")
            print(pad_deg)
            
            pad_deg = self.tanh(pad_deg)

            print("===================== Pad deg tanh1 =================")
            print(pad_deg)
            #=====
            pad_deg = self.max_pooling(pad_deg)
                
            print("===================== Pad deg max_pooling=================")
            print(pad_deg)

            pad_deg = pad_deg.squeeze(1)

            print("===================== Pad deg sequeeze=================")
            print(pad_deg)
                
            pad_deg = self.linear_5(pad_deg)

            print("===================== Pad deg linear_5=================")
            print(pad_deg)
            
            pad_deg = self.tanh(pad_deg)
                
            print("===================== Pad deg tanh2 =================")
            print(pad_deg)


            #doc_vectors = F.dropout(self.bn(F.relu(self.hidden_layer(lstm_out))), p=self.dropout, training=self.training)
            #deg_vectors = F.dropout(self.bn(F.relu(self.hidden_layer(deg_vec))), p=self.dropout, training=self.training)
            print("========================= Pad deg =======================")
            print(pad_deg)
            global_deg_vec.append(pad_deg)
            print("========================= Global Pad deg =======================")
            print(global_deg_vec)