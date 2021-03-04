
import argparse
import logging
import os
import sys
import torch
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from model_def import ElectraClassifier, DualElectra
from utils import save_model
from data_prep import get_data_loader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def train(args):

    model = DualElectra(args.model_checkpoint,args.num_labels)

    # Setting up cuda 
    use_cuda = args.num_gpus > 0
    if use_cuda:
        device='cuda:0'
        torch.cuda.manual_seed(args.seed)
        if args.num_gpus > 1:
            model = torch.nn.DataParallel(model)
        model.cuda()
    else:
        device='cpu'
        torch.manual_seed(args.seed)
    print('got model')
    # tokenizer,dataloader and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)
    train_path = os.path.join(args.data_dir,args.train)
    if use_cuda:
        num_workers = args.num_gpus * 4
    else:
        num_workers = args.num_cpus
    
    print('started getting train_loader')
    train_loader,train_data = get_data_loader(train_path,tokenizer,args.max_len,args.batch_size,num_workers)

    # Setting the optimizer (Important that this is done after, and not before, moving the model to cuda)
    optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr = args.lr, 
            eps = args.epsilon,
            weight_decay=args.weight_decay)


    print('starting to train')
    #loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.,3.])).to(device)
    
    # weights = [ 0.8, #Race
    #             0.8, #Religion
    #             1.0, #Seksuel orientering
    #             0.8, #Køn
    #             1.0, #Socialklasse
    #             0.8, #Politisk orientering
    #             0.8, #Handicap
    #             1.0, #Alder
    #             0.6, #Offentlig
    #             0.8, #ikke-offentlig
    #             0.8, #ikke-beskyttet gruppe,
    #             1.0, #anden gruppe
    #             ]
    # weights = torch.FloatTensor(weights).to(device)
    # loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Train
    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0
        correct = 0
        print('Epoch', epoch)
        for step, batch in enumerate(train_loader):
            if args.verbose:
                print(step)

            b_input_ids_text = batch['input_ids_text'].to(device)
            b_input_mask_text = batch['attention_mask_text'].to(device)
            b_input_ids_context = batch['input_ids_context'].to(device)
            b_input_mask_context = batch['attention_mask_context'].to(device)
            b_labels = batch['targets'].to(device)

            logits = model(b_input_ids_text, attention_mask_text=b_input_mask_text, input_ids_context=b_input_ids_context, attention_mask_context=b_input_mask_context)
            loss = loss_fn(logits.view(-1, args.num_labels), b_labels.view(-1))
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            running_loss += loss.item() * b_input_ids_text.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == b_labels).sum().item()

        running_loss = running_loss/train_data.__len__()
        running_accuracy = 100*(correct/train_data.__len__())
        print('Running loss', running_loss)
        print('Running accuracy', running_accuracy)

    # Test on eval data
    eval_path = os.path.join(args.data_dir,args.valid)
    eval_loader,valid_data = get_data_loader(eval_path,tokenizer,args.max_len,args.test_batch_size,num_workers)
    test(model, eval_loader, device)

    ## save model
    if args.save_model:
        save_model(model, args.model_dir,args.num_gpus)

def test(model, eval_loader, device):
    model.eval()
    predicted_classes = torch.empty(0).to(device)
    labels = torch.empty(0).to(device)

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            if args.verbose:
                print(step)
            b_input_ids_text = batch['input_ids_text'].to(device)
            b_input_mask_text = batch['attention_mask_text'].to(device)
            b_input_ids_context = batch['input_ids_context'].to(device)
            b_input_mask_context = batch['attention_mask_context'].to(device)
            b_labels = batch['targets'].to(device)

            logits = model(b_input_ids_text, attention_mask_text=b_input_mask_text, input_ids_context=b_input_ids_context, attention_mask_context=b_input_mask_context)
            _,preds = torch.max(logits, dim=1)

            predicted_classes = torch.cat((predicted_classes, preds))
            labels = torch.cat((labels, b_labels))

    predicted_classes = predicted_classes.to('cpu')
    labels = labels.to('cpu')

    print("confusion matrix:")
    print(confusion_matrix(labels, predicted_classes))
    print('F1 score:', f1_score(labels, predicted_classes, average='weighted'))
    print('Precision score:', precision_score(labels, predicted_classes, average='weighted'))
    print('Recall score:', recall_score(labels, predicted_classes, average='weighted'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--model-checkpoint", type=str, default='Maltehb/-l-ctra-danish-electra-small-cased', help="name of pretrained model from huggingface model hub")
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--train", type=str, default='train.csv')
    parser.add_argument("--valid", type=str, default='valid.csv')
    parser.add_argument("--test", type=str, default='test.csv')
    # Hyperparams
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--test-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs to train (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--epsilon", type=int, default=1e-8)  
    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--num-cpus", type=int, default=os.environ["SM_NUM_CPUS"])
    parser.add_argument("--save-model", type=int, default=1)
    # Log
    parser.add_argument("--verbose", type=bool, default=False)

    ## RUN
    args = parser.parse_args()
    train(args)