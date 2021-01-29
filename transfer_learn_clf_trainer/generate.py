import os
import json
import torch
import csv

from io import StringIO
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Maltehb/-l-ctra-danish-electra-small-uncased", use_fast=True)

JSON_CONTENT_TYPE = 'application/json'


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
                
        encoded_data = tokenizer(data['text'], return_tensors='pt')
        
        input_id = encoded_data['input_ids']
        input_mask = encoded_data['attention_mask']

        print("================ encoded sentences ==============")
        
        print(input_id)
        
        print("================= padded input and attention mask ================")
        print(input_id, '\n', input_mask)

        return input_id, input_mask
    elif request_content_type == 'text/csv':
        # Read the raw input data as CSV.

        data_list = []

        #df = pd.read_csv(StringIO(serialized_input_data), 
        #                 header=None, sep='\t')

        f = open(StringIO(serialized_input_data), newline='')
        reader = csv.reader(f, delimiter='\t')
        for i in reader:
            data_list.append(i[1])
        f.close()

        encoded_data = tokenizer(data_list, return_tensors='pt', padding=True)

        input_id = encoded_data['input_ids']
        input_mask = encoded_data['attention_mask']
        
        return input_id, input_mask
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
        #print(y)
        probs = y.softmax(1).tolist()
        #print(probs)
    return probs

