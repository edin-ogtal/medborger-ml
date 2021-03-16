import argparse
import logging
import os
import sys
import torch
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from model_def import ElectraWithContextClassifier
from data_prep import get_data_with_context_loader

def get_model(model_checkpoint, model_weights, num_labels):
    model = ElectraWithContextClassifier(model_checkpoint,num_labels)
    model_path = os.path.join(args.model_dir,model_weights)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return(model)


def train(args):

    model = get_model(args.model_checkpoint, args.model_weights, args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)

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

    if use_cuda:
        num_workers = args.num_gpus * 4
    else:
        num_workers = args.num_cpus
    
    # Test on eval data
    eval_path = os.path.join(args.data_dir,args.valid)
    eval_loader,_ = get_data_with_context_loader(eval_path,tokenizer,args.max_len,args.test_batch_size,num_workers)
    test(model, eval_loader, device)


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
    parser.add_argument("--model-weights", type=str, default='pytorch_model_from_owncloud.bin')

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