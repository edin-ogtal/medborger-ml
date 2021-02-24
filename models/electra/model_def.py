
#from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(512, 200)
        self.relu1 = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(200, 100)
        self.relu2 = torch.nn.ReLU()
        self.net3 = torch.nn.Linear(100, 2)

    def forward(self, x):
        x = self.relu1(self.net1(x))
        x = self.relu2(self.net2(x))
        return self.net3(x)


from transformers import AutoTokenizer, AutoModel
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.functional as F
import torch.nn as nn


model_checkpoint = 'Maltehb/-l-ctra-danish-electra-small-uncased'
class TextClassifier(nn.Module):
    def __init__(self, num_labels):
        super(TextClassifier, self).__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_checkpoint)
        self.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                        nn.Linear(256, num_labels),
                                        nn.ReLU())
        
    def forward(self, input_ids, attention_mask):
      output = self.pretrained_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
      )
      hidden_state = output.last_hidden_state
      #pooler = hidden_state[:, 0]
      output = self.classifier(hidden_state)      
      return output


# model = ToyModel()


# from train import _get_train_data_loader
# train_loader = _get_train_data_loader(16, '.')

# for batch in train_loader:
#     break
# batch['input_ids'] = batch['input_ids'].type(torch.FloatTensor)
# output = model(batch['input_ids'])

# class CNNClassifier(nn.Module):
    
#     def __init__(self,pretrained_model_name,n_classes,max_len=512,freeze_bert=True,bert_layers_to_freeeze=10,n_kernels=3,kernel_sizes=[2,3,4],dropout=0.2):
#         super(CNNClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained(pretrained_model_name)
#         self.freeze_bert = freeze_bert
#         self.bert_layers_to_freeeze = bert_layers_to_freeeze

#         self.convs = nn.ModuleList([nn.Conv2d(1, n_kernels, (k_size, max_len)) for k_size in kernel_sizes])
#         self.dropout = nn.Dropout(dropout)
#         self.fc1 = nn.Linear(len(kernel_sizes) * n_kernels, n_classes)
#         # self.softmax = nn.Softmax(1)

#     def classifier(self,bert_embd):

#         embd = bert_embd.unsqueeze(1)
#         embd = torch.transpose(embd,2,3)
#         # DO CONV
#         embd = [F.relu(conv(embd)).squeeze(3) for conv in self.convs]
#         embd = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in embd]         
#         embd = torch.cat(embd, 1)

#         # dropout,linear,softmax
#         embd = self.dropout(embd)
#         logits = self.fc1(embd)
#         # probs = self.softmax(logits) # it looks like crossEntropy do this part itself
#         return logits

#     # def forward(self, x):
#     #     probs = self.classifier(x)
#     #     return probs

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(
#           input_ids=input_ids,
#           attention_mask=attention_mask
#         )
#         if self.freeze_bert:
#             freeze(self.bert,self.bert_layers_to_freeeze)
#         probs = self.classifier(outputs[0]) # takes a tensor of shape [batch_size,max_len,bert_embd_size]
#         return probs

# # MAX_LEN = 512
# # a = torch.randn(16,512)
# # model = CNNClassifier(PRETRAINED_MODEL_NAME,2,MAX_LEN)  
# # p = model(a)
# # print(p)


