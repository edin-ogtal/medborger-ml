import pandas as pd
import torch
from torch.utils.data import Dataset,RandomSampler,DataLoader,SequentialSampler
import numpy as np
import time
import argparse
import os
from transformers import AutoTokenizer

from model_def import ElectraWithContextClassifier

class DualDatasetInference(Dataset):
    def __init__(self, text, context, ids, tokenizer, max_len):
        self.text = text
        self.context = context
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        context = self.context[item]
        id = self.ids[item]
        tokenized_text = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        input_ids_text = tokenized_text['input_ids'].squeeze()
        attention_mask_text = tokenized_text['attention_mask'].squeeze()

        tokenized_context = self.tokenizer(
            context,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        input_ids_context = tokenized_context['input_ids'].squeeze()
        attention_mask_context = tokenized_context['attention_mask'].squeeze()

        return {
          'text': text,
          'input_ids_text': input_ids_text,
          'attention_mask_text': attention_mask_text,
          'input_ids_context': input_ids_context,
          'attention_mask_context': attention_mask_context,
          'id': id,

        }


def get_inference_loader(data_dir, data_file,tokenizer,max_len,batch_size,num_workers):
    dataset = pd.read_csv(os.path.join(data_dir, data_file), sep='\t', names = ['ids', 'origin', 'text', 'main_text', 'secondary_text'])

    dataset['context'] = dataset.origin.astype(str) + ' \n ' + dataset.main_text.astype(str) + ' \n ' + dataset.secondary_text.astype(str)
    dataset['text'] = dataset.text.astype(str)
    dataset['ids'] = dataset.ids.astype(str)
    data = DualDatasetInference(
                    text=dataset.text.to_numpy(),
                    context=dataset.context.to_numpy(),
                    ids=dataset.ids.to_numpy(),
                    tokenizer=tokenizer,
                    max_len=max_len
                    )

    #sampler = RandomSampler(data)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data,batch_size=batch_size,sampler=sampler,num_workers=num_workers,pin_memory=True)
    return dataloader,data



def get_model(model_checkpoint, model_weights, num_labels):
    model = ElectraWithContextClassifier(model_checkpoint,num_labels)
    model_path = os.path.join(args.model_dir,model_weights)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return(model)


def do_inference(args):

    model = get_model(args.model_checkpoint, args.model_weights, args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)

    train_loader,train_data = get_inference_loader(args.data_dir,args.input_data,tokenizer,args.max_len,args.batch_size,args.num_cpus)

    t_start = time.time()

    result_ids = []
    results = {}

    with torch.no_grad():
        for step, batch in enumerate(train_loader):
            
            b_input_ids_text = batch['input_ids_text']
            b_input_mask_text = batch['attention_mask_text']
            b_input_ids_context = batch['input_ids_context']
            b_input_mask_context = batch['attention_mask_context']
            b_ids = batch['id']

            logits = model(b_input_ids_text, attention_mask_text=b_input_mask_text, input_ids_context=b_input_ids_context, attention_mask_context=b_input_mask_context)
            y = logits.softmax(1).tolist()

            for i, idx in enumerate(b_ids):
                results[idx] = y[i]

            if args.verbose:
                print(f'Finished {(step+1)*4} samples after {round(time.time() - t_start, 2)} seconds. {round((step+1)*4 / (round(time.time() - t_start, 2)), 2)} samples pr. second')
            else:
                if step % 100 == 0:
                    print(f'Finished {(step+1)*4} samples after {round(time.time() - t_start, 2)} seconds. {round((step+1)*4 / (round(time.time() - t_start, 2)), 2)} samples pr. second')

    df = pd.DataFrame.from_dict(results)

    df.to_csv(args.output_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--model-checkpoint", type=str, default='Maltehb/-l-ctra-danish-electra-small-cased', help="name of pretrained model from huggingface model hub")
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--model-weights", type=str, default='pytorch_model.bin')
    
    # Hyperparams
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--num-cpus", type=int, default=os.environ["SM_NUM_CPUS"])
    parser.add_argument("--input-data", type=str, default='dkmed_rep_2019_march_may.csv')
    parser.add_argument('--output-data', type=str, default='output.csv')

    # Log
    parser.add_argument("--verbose", type=bool, default=False)

    ## RUN
    args = parser.parse_args()
    do_inference(args)