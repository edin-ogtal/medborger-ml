import argparse
import logging
import os

import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from datasets import load_dataset, load_metric

def get_model(model_checkpoint, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    return(model)
    
    
def get_tokenizer(model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    return(tokenizer)


def get_encoded_data(data_loader_script, tokenizer):
    dataset = load_dataset(data_loader_script)

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512) 
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    return(encoded_dataset)


def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    tokenizer = get_tokenizer(args.model_checkpoint)
    model = get_model(args.model_checkpoint, args.num_labels)

    encoded_dataset = get_encoded_data('create_dataset.py', tokenizer)

    metric = load_metric("glue", "mrpc")
    
    metric_name = "accuracy"

    train_args = TrainingArguments(
        "test-glue",
        evaluation_strategy = "epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis = 1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model, 
        train_args, 
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print(trainer.predict(encoded_dataset['test']).metrics)

    print('started saving')

    trainer.save_model(args.model_dir)

    print('done saving')



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
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-5, metavar="LR", help="learning rate (default: 0.3e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.01, metavar="M", help="weight_decay (default: 0.01)")
    parser.add_argument("--seed", type=int, default=43, metavar="S", help="random seed (default: 43)")
    parser.add_argument("--epsilon", type=int, default=1e-8, metavar="EP", help="random seed (default: 1e-8)")
    #parser.add_argument("--frozen_layers", type=int, default=10, metavar="NL", help="number of frozen layers(default: 10)")
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
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    #parser.add_argument("--data-dir", type=str, default='.')

    #parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    #parser.add_argument("--num-gpus", type=int, default=False)


    args = parser.parse_args()

    train(args)
