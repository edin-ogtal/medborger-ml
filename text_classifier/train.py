import argparse
import logging
import os
import sys
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch_optimizer as optim

from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from datasets import load_dataset, load_metric

from torch.utils.data import RandomSampler, DataLoader

# Network definition
from data_prep import MedborgerDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MAX_LEN = 512
PRETRAINED_MODEL_NAME = 'Maltehb/-l-ctra-danish-electra-small-uncased'
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, use_fast=True)

def _get_train_data_loader(batch_size, data_dir):
    dataset = pd.read_csv(os.path.join(args.data_dir, 'train.csv'), sep='\t', names = ['targets', 'text'])
    train_data = MedborgerDataset(
                    text=dataset.text.to_numpy(),
                    targets=dataset.targets.to_numpy(),
                    tokenizer=tokenizer,
                    max_len=MAX_LEN
                    )

    #Maybe use different sampler???
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    return(train_dataloader)


def _get_eval_data_loader(batch_size, data_dir):
    dataset = pd.read_csv(os.path.join(args.data_dir, 'valid.csv'), sep='\t', names = ['targets', 'text'])
    train_data = MedborgerDataset(
                    text=dataset.text.to_numpy(),
                    targets=dataset.targets.to_numpy(),
                    tokenizer=tokenizer,
                    max_len=MAX_LEN
                    )
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    return(train_dataloader)

#def freeze(model, frozen_layers):
#    modlu


def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)


    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

def get_model(model_checkpoint, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    return(model)

def get_encoded_data(data_loader_script, tokenizer):
    dataset = load_dataset(data_loader_script)

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=MAX_LEN) 
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    return(encoded_dataset)


def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    model = get_model(args.model_checkpoint, args.num_labels)

    #encoded_dataset = get_encoded_data('create_dataset.py', tokenizer)
    
    # Maybe use different optimizer????
    optimizer = optim.Lamb(
            model.parameters(), 
            lr = args.lr, 
            betas=(0.9, 0.999), 
            eps=args.epsilon, 
            weight_decay=args.weight_decay)

    # Maybe use different loss function
    loss_fn = nn.CrossEntropyLoss().to(device)


    for epoch in range(1, args.epochs + 1):
        model.train()

        for step, batch in enumerate(train_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            outputs = model(b_input_ids,attention_mask=b_input_mask)

            loss = loss_fn(outputs.logits, b_labels)

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) ?????
            optimizer.step()
            optimizer.zero_grad()

            if args.verbose:
                print('Batch', step)

    model.save_pretrained(args.model_dir)

    eval_loader = _get_eval_data_loader(args.test_batch_size, args.test)        
    test(model, eval_loader, device)



def test(model, eval_loader, device):
    model.eval()
    predicted_classes = torch.empty(0)
    labels = torch.empty(0)

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            print(step)
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            outputs = model(b_input_ids,attention_mask=b_input_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predicted_classes = torch.cat((predicted_classes, preds))
            labels = torch.cat((labels, b_labels))

    print("confusion matrix:")
    print(confusion_matrix(labels, predicted_classes))
    print('F1 score:', f1_score(labels, predicted_classes))
    print('Precision score:', precision_score(labels, predicted_classes))
    print('Recall score:', recall_score(labels, predicted_classes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--model_checkpoint", type=str, default='Maltehb/-l-ctra-danish-electra-small-uncased', help="name of pretrained model from huggingface model hub"
    )
    parser.add_argument(
        "--num_labels", type=int, default=2, metavar="N", help="Number of labels."
    )

    parser.add_argument(
        "--batch-size", type=int, default=16, metavar="N", help="input batch size for training (default: 16)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=8, metavar="N", help="input batch size for testing (default: 8)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-5, metavar="LR", help="learning rate (default: 0.3e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01, metavar="M", help="weight_decay (default: 0.01)")
    parser.add_argument("--seed", type=int, default=43, metavar="S", help="random seed (default: 43)")
    parser.add_argument("--epsilon", type=int, default=1e-8, metavar="EP", help="random seed (default: 1e-8)")
    #parser.add_argument("--frozen_layers", type=int, default=10, metavar="NL", help="number of frozen layers(default: 10)")
    parser.add_argument('--verbose', default=True,help='For displaying logs')
    #parser.add_argument(
    #    "--log-interval",
    #    type=int,
    #    default=10,
    #    metavar="N",
    #    help="how many batches to wait before logging training status",
    #)

    # Container environment
    #parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    #parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    #parser.add_argument("--data-dir", type=str, default='.')

    #parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    #parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--num-gpus", type=int, default=False)


    args = parser.parse_args()

    train(args)
