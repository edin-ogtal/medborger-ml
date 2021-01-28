import json
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Maltehb/-l-ctra-danish-electra-small-uncased", use_fast=True)

#import json
#import logging
#import os

#import torch
#from rnn import RNNModel

#import data

#JSON_CONTENT_TYPE = 'application/json'

#logger = logging.getLogger(__name__)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("================ objects in model_dir ===================")
    print(os.listdir(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    print("================ model loaded ===========================")
    return model.to(device)

def input_fn(serialized_input_data, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":

        MAX_LEN = 512

        data = json.loads(serialized_input_data)
        print("================ input sentences ===============")
        print(data)
        
        if isinstance(data, str):
            data = [data]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            pass
        else:
            raise ValueError("Unsupported input type. Input type can be a string or an non-empty list. \
                             I got {}".format(data))
                       
        #encoded = [tokenizer.encode(x, add_special_tokens=True) for x in data]
        #encoded = tokenizer(data, add_special_tokens=True) 
        
        # for backward compatibility use the following way to encode 
        # https://github.com/huggingface/transformers/issues/5580
        input_ids = [tokenizer.encode(x, add_special_tokens=True) for x in data]
        
        print("================ encoded sentences ==============")
        print(input_ids)

        # pad shorter sentence
        padded =  torch.zeros(len(input_ids), MAX_LEN) 
        for i, p in enumerate(input_ids):
            padded[i, :len(p)] = torch.tensor(p)
     
        # create mask
        mask = (padded != 0)
        
        print("================= padded input and attention mask ================")
        print(padded, '\n', mask)

        return padded.long(), mask.long()
    raise ValueError("Unsupported content type: {}".format(request_content_type))


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    #logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)



def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_id, input_mask = input_data
    input_id = input_id.to(device)
    input_mask = input_mask.to(device)
    print("============== encoded data =================")
    print(input_id, input_mask)
    with torch.no_grad():
        y = model(input_id, attention_mask=input_mask)[0]
        print("=============== inference result =================")
        print(y)
    return y

