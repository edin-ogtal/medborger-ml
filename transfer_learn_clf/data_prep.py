import torch
#from torch import nn
#import torch.utils.data
#import torch.utils.data.distributed
from torch.utils.data import Dataset

class MedborgerDataset(Dataset):
    def __init__(self, sentence, label, tokenizer, max_len):
        self.sentence = sentence
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, item):
        sentence = str(self.sentence[item])
        label = self.label[item]
        encoding = self.tokenizer.encode_plus(
            sentence,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
          'sentence': sentence,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'label': torch.tensor(label, dtype=torch.long)
        }