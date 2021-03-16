from transformers import ElectraModel, AutoModel
import torch
import torch.nn.functional as F
import torch.nn as nn


class ElectraWithContextClassifier(nn.Module):
    
    def __init__(self,pretrained_model_name,num_labels=2):
        super(ElectraWithContextClassifier, self).__init__()
        self.num_labels = num_labels
        self.electra_text = ElectraModel.from_pretrained(pretrained_model_name)
        self.electra_context = ElectraModel.from_pretrained(pretrained_model_name)

        self.avg_pool = nn.AvgPool1d(2, 2)
        self.dense = nn.Linear(self.electra_text.config.hidden_size*2, self.electra_text.config.hidden_size*2)
        self.dropout = nn.Dropout(self.electra_text.config.hidden_dropout_prob)
        self.out_proj = nn.Linear(self.electra_text.config.hidden_size*2, self.num_labels)

    def process_hidden_states(self,hs_text,hs_context):

        hs_text = hs_text[:, 0, :]
        hs_context = hs_context[:, 0, :]
        # take mean of all hidden layers. unsqueeze it, and reduce the len of the embedding with AvgPool1d with kernel_size and stride as 2
        # hs_text = self.avg_pool(hs_text.unsqueeze(dim=1)).squeeze(dim=1)
        # hs_context = self.avg_pool(hs_context.unsqueeze(dim=1)).squeeze(dim=1)
        # concat the layers
        return torch.cat((hs_text,hs_context),dim=1)

    def classifier(self,x):
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        x = F.gelu(self.dense(x))
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits

    def forward(self, input_ids_text=None,attention_mask_text=None,input_ids_context=None,attention_mask_context=None):

        hs_text = self.electra_text(input_ids=input_ids_text,attention_mask=attention_mask_text)[0]
        hs_context = self.electra_context(input_ids=input_ids_context,attention_mask=attention_mask_context)[0]
        hs_combined = self.process_hidden_states(hs_text,hs_context)
        logits = self.classifier(hs_combined)
        return logits