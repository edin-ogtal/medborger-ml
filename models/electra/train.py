import argparse
import logging
import time
import os
import sys
import torch
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
from tqdm import tqdm

from model_def import ElectraClassifier
from utils import save_model
from data_prep import get_data_loader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def train(args):

    torch.manual_seed(args.seed)
    model = ElectraClassifier(args.model_checkpoint,args.num_labels)

    # Setting up cuda
    if args.num_gpus > 0:
        device = 'cuda:0'
    else:
        device = 'cpu'

    for param in model.electra.parameters():
        param.requires_grad = False

    model = model.to(device)
    if args.num_cpus > 1:
        model = torch.nn.DataParallel(model)

    # tokenizer,dataloader and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    train_path = os.path.join(args.data_dir,args.train)
    train_loader,train_data = get_data_loader(train_path,tokenizer,args.max_len,args.batch_size)

    # Setting the optimizer (Important that this is done after, and not before, moving the model to cuda)
    optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr = args.lr, 
            eps = args.epsilon,
            weight_decay=args.weight_decay)

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.01,1.0,1.0])).to(device)
    #loss_fn = torch.nn.CrossEntropyLoss().to(device)
   
    # Train
    model.train()
    # print(torch.cuda.memory_reserved())

    for epoch in range(1, args.epochs + 1):
        e_start = time.time()
        running_loss = 0
        correct = 0
        print('Epoch', epoch)
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True):
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            if args.use_half_precision:
                with torch.cuda.amp.autocast():
                    logits = model(b_input_ids, attention_mask=b_input_mask)   
                    loss = loss_fn(logits.view(-1, args.num_labels), b_labels.view(-1))
            else:
                logits = model(b_input_ids, attention_mask=b_input_mask)   
                loss = loss_fn(logits.view(-1, args.num_labels), b_labels.view(-1))

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            running_loss += loss.item() * b_input_ids.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == b_labels).sum().item()

        running_loss = running_loss/train_data.__len__()
        running_accuracy = 100*(correct/train_data.__len__())
        print('Running loss', running_loss)
        print('Running accuracy', running_accuracy)
        print(f'Finished epoch after {round(time.time() - e_start, 2)} seconds.')

    ## save model
    if args.save_model:
        save_model(model, args.model_dir,args.num_gpus)

    # Test on eval data
    eval_path = os.path.join(args.data_dir,args.eval)
    eval_loader,_ = get_data_loader(eval_path,tokenizer,args.max_len,args.test_batch_size)
    predictions,true_labels,texts = test(model, eval_loader,device)

    # Export predictions to csv
    obs = list()
    for pred,true,text in zip(predictions,true_labels,texts):
        d = {'predicted_label':pred,'true_label':true,'correct':true==pred,'text':text}
        obs.append(d)
    df = pd.DataFrame(obs)
    data_path = os.path.join(args.data_dir,'predicted.csv')
    df.to_csv(data_path,encoding='utf-8',sep='\t')

def test(model, eval_loader,device):
    model.eval()
    predictions = torch.empty(0,device=device)
    true_labels = torch.empty(0,device=device)
    texts = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), position=0, leave=True):
            texts += batch['text']
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['targets'].to(device)

            if args.use_half_precision:
                with torch.cuda.amp.autocast():
                    logits = model(b_input_ids, attention_mask=b_input_mask)   
            else:
                logits = model(b_input_ids, attention_mask=b_input_mask)   

            _,preds = torch.max(logits, dim=1)

            predictions = torch.cat((predictions, preds))
            true_labels = torch.cat((true_labels, b_labels))

    predictions = predictions.detach().cpu().numpy()
    true_labels = true_labels.detach().cpu().numpy()
    texts = np.asarray(texts)

    print("confusion matrix:")
    print(confusion_matrix(true_labels, predictions))
    print('F1 score:', f1_score(true_labels, predictions,average='macro'))
    print('Precision score:', precision_score(true_labels, predictions,average='macro'))
    print('Recall score:', recall_score(true_labels, predictions,average='macro'))
    return predictions,true_labels,texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--model-checkpoint", type=str, default='Maltehb/-l-ctra-danish-electra-small-cased', help="name of pretrained model from huggingface model hub")
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--train", type=str, default='train.csv')
    parser.add_argument("--eval", type=str, default='eval.csv')
    parser.add_argument("--test", type=str, default='test.csv')
    # Hyperparams
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epsilon", type=float, default=1e-8)

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--num-cpus", type=int, default=os.environ["SM_NUM_CPUS"])
    parser.add_argument("--save-model", type=int, default=1)

    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--use-half-precision", action='store_true')

    ## RUN
    args = parser.parse_args()
    train(args)