from pathlib import Path

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        print(label_dir)
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

train_texts, train_labels = read_imdb_split('aclImdb/train')
test_texts, test_labels = read_imdb_split('aclImdb/test')

n = 3*16

train_texts = train_texts[0:n] + train_texts[-n:0]
train_labels = train_labels[0:n] + train_labels[-n:0]

test_texts = test_texts[0:n] + test_texts[-n:]
test_labels = test_labels[0:n] + test_labels[-n:]

from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)


from transformers import DistilBertTokenizerFast, AutoTokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
#tokenizer = AutoTokenizer.from_pretrained("Maltehb/-l-ctra-danish-electra-small-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)



class DanishELECTRA(torch.nn.Module):
    def __init__(self):
        super(DanishELECTRA, self).__init__()
        self.l1 = AutoModelForPreTraining.from_pretrained("Maltehb/-l-ctra-danish-electra-small-uncased")
        self.pre_classifier = torch.nn.Linear(256, 256)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output



#model = DanishELECTRA()
#device='cpu'
#model.to(device)




class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = AutoModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, AutoModelWithLMHead, AutoModel, AutoModelForPreTraining, AutoModelForSequenceClassification

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=1,
)


model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
#model = DistilBertForSequenceClassification.from_pretrained("Maltehb/-l-ctra-danish-electra-small-uncased")
#model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset  
    test_dataset=test_dataset,
    compute_metrics = compute_metrics 
)

trainer.train()


print(trainer.evaluate())



#LEARNING_RATE = 0.1


# loss_function = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


# def calcuate_accu(big_idx, targets):
#     n_correct = (big_idx==targets).sum().item()
#     return n_correct


# def train(epoch):
#     tr_loss = 0
#     n_correct = 0
#     nb_tr_steps = 0
#     nb_tr_examples = 0
#     model.train()

#     for _,data in enumerate(train_dataset):
#         ids = data['input_ids'].to(device, dtype = torch.long)
#         mask = data['attention_mask'].to(device, dtype = torch.long)
#         targets = data['labels'].to(device, dtype = torch.long)

#         outputs = model(ids, mask)
#         loss = loss_function(outputs, targets)
#         tr_loss += loss.item()
#         big_val, big_idx = torch.max(outputs.data, dim=1)
#         n_correct += calcuate_accu(big_idx, targets)

#         nb_tr_steps += 1
#         nb_tr_examples+=targets.size(0)
        
#         if _%5==0:
#             loss_step = tr_loss/nb_tr_steps
#             accu_step = (n_correct*100)/nb_tr_examples 
#             print(f"Training Loss per 5000 steps: {loss_step}")
#             print(f"Training Accuracy per 5000 steps: {accu_step}")

#         optimizer.zero_grad()
#         loss.backward()
#         # # When using GPU
#         optimizer.step()

#     print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
#     epoch_loss = tr_loss/nb_tr_steps
#     epoch_accu = (n_correct*100)/nb_tr_examples
#     print(f"Training Loss Epoch: {epoch_loss}")
#     print(f"Training Accuracy Epoch: {epoch_accu}")

#     return
# from torch.utils.data import DataLoader
# from transformers import DistilBertForSequenceClassification, AdamW

# #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device("cpu")

# model = DistilBertForSequenceClassification.from_pretrained('sshleifer/tiny-distilbert-base-cased')
# model.to(device)
# model.train()

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# optim = AdamW(model.parameters(), lr=5e-5)

# for epoch in range(1):
#     for batch in train_loader:
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs[0]
#         loss.backward()
#         optim.step()

# model.eval()
