import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from transformers import BertForSequenceClassification

class BERTSem(nn.Module):

  def __init__(self):
    super(BERTSem, self).__init__()
    self.bert_layer = BertForSequenceClassification.from_pretrained(
          "bert-base-uncased", 
          num_labels = 3,  
          output_attentions = False,
          output_hidden_states = False,
      )
    # self.bert_layer = self.bert_layer

  def forward(self, b_input_ids, b_input_mask, b_labels=None):
    return self.bert_layer(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True)

# Tell pytorch to run this model on the GPU.
