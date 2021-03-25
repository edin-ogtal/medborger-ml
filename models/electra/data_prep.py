import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,RandomSampler,DataLoader

class MedborgerDataset(Dataset):
    def __init__(self, text, targets, tokenizer, max_len):
        self.text = text
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        target = self.targets[item]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt',
        )
        return {
          'text': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target)
        }


def get_data_loader(path,tokenizer,max_len,batch_size):
    dataset = pd.read_csv(path, sep='\t', names = ['targets', 'target_names', 'text', 'origin', 'main_text', 'secondary_text', 'source'])
    dataset = remove_invalid_inputs(dataset,'text')

    data = MedborgerDataset(
                    text=dataset.text.to_numpy(),
                    targets=dataset.targets.to_numpy(),
                    tokenizer=tokenizer,
                    max_len=max_len
                    )

    sampler = RandomSampler(data)
    dataloader = DataLoader(data,batch_size=batch_size,sampler=sampler,pin_memory=True)
    return dataloader,data


def remove_invalid_inputs(dataset,text_column):
    'Simpel metode til at fjerne alle rækker fra en dataframe, baseret på om værdierne i en kolonne er af typen str'
    dataset['valid'] = dataset[text_column].apply(lambda x: isinstance(x, str))
    return dataset.loc[dataset.valid]
