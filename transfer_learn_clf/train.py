import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import torch_optimizer as optim

# Network definition
from clf_model import TextClassifier
from data_prep import MedborgerDataset
 
## SageMaker Distributed code.
#from smdistributed.dataparallel.torch.parallel.distributed import DistributedDataParallel as DDP
#import smdistributed.dataparallel.torch.distributed as dist

#dist.init_process_group()


MAX_LEN = 512  # this is the max length of the sequence
model_checkpoint = 'Maltehb/-l-ctra-danish-electra-small-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def _get_train_data_loader(batch_size, training_dir):
    dataset = pd.read_csv(os.path.join(training_dir, "train_df.csv"), sep="\t")
    train_data = MedborgerDataset(
        sentence=dataset.sentence.to_numpy(),
        label=dataset.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    return train_dataloader

def _get_test_data_loader(batch_size, training_dir):
    dataset = pd.read_csv(os.path.join(training_dir, "test_df.csv"), sep="\t")
    test_data = MedborgerDataset(
        sentence=dataset.sentence.to_numpy(),
        label=dataset.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
  )
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    return test_dataloader


def freeze(model, frozen_layers):
    for param in model.pretrained_model.electra.parameters():
        param.requires_grad = False




def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir)

    print(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    model = TextClassifier(
        args.num_labels  # The number of output labels.
    )

    freeze(model, args.frozen_layers)
    
    optimizer = optim.Lamb(
            model.parameters(), 
            lr = args.lr, 
            betas=(0.9, 0.999), 
            eps=args.epsilon, 
            weight_decay=args.weight_decay)
    
    total_steps = len(train_loader.dataset)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps)
    
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['label'].to(device)

            #outputs = model(b_input_ids,attention_mask=b_input_mask)
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs.loss
            #loss = loss_fn(outputs, b_labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            optimizer.zero_grad()
            
            print(
                "Collecting data from Master Node: \n Train Epoch: {} [{}/{} ({:.0f}%)] Training Loss: {:.6f}".format(
                    epoch,
                    step * len(batch['input_ids']),
                    len(train_loader.dataset),
                    100.0 * step / len(train_loader),
                    loss.item(),
                    )
            )
            test(model, test_loader, device)
            print('Batch', step)
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
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
    parser.add_argument("--lr", type=float, default=0.3e-5, metavar="LR", help="learning rate (default: 0.3e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01, metavar="M", help="weight_decay (default: 0.01)")
    parser.add_argument("--seed", type=int, default=43, metavar="S", help="random seed (default: 43)")
    parser.add_argument("--epsilon", type=int, default=1e-8, metavar="EP", help="random seed (default: 1e-8)")
    parser.add_argument("--frozen_layers", type=int, default=10, metavar="NL", help="number of frozen layers(default: 10)")
    #parser.add_argument('--verbose', action='store_true', default=False,help='For displaying SMDataParallel-specific logs')
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
    #parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    #parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--data-dir", type=str, default='.')

    #parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    #parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--num-gpus", type=int, default=False)

    train(parser.parse_args())


