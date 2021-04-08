from transformers import BertModel
import torch
import torch.nn.functional as F
import torch.nn as nn

class BertClassifier(nn.Module):
    
    def __init__(self,pretrained_model_name,num_labels=2):
        super(BertClassifier, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, self.num_labels)


    def forward(self, input_ids=None,attention_mask=None):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        #print(_)
        #print(pooled_output)
        output = self.drop(pooled_output)
        return self.out(output)
