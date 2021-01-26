from transformers import AutoTokenizer, AutoModel
#import torch
#import torch.nn.functional as F
import torch.nn as nn


model_checkpoint = 'Maltehb/-l-ctra-danish-electra-small-uncased'
class TextClassifier(nn.Module):
    def __init__(self, num_labels):
        super(TextClassifier, self).__init__()
        self.pretrained_lm = AutoModel.from_pretrained(model_checkpoint, num_labels=num_labels)
        self.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                        nn.Linear(self.pretrained_lm.config.hidden_size, num_labels),
                                        nn.Tanh())
        
    def forward(self, input_ids, attention_mask, labels):
      output = self.pretrained_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels = labels,
      )
      return output
