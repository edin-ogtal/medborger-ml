import os
import json
import torch
import csv

from io import StringIO
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from model_def import ElectraClassifier, DualElectra, ElectraWithContextClassifier

MAX_LEN = 512  # this is the max length of the sequence
PRE_TRAINED_MODEL_NAME = "Maltehb/-l-ctra-danish-electra-small-uncased"

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, use_fast=True)

JSON_CONTENT_TYPE = 'application/json'


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(model_dir)
    #print("================ objects in model_dir ===================")
    #print(os.listdir(model_dir))
    model = ElectraWithContextClassifier(PRE_TRAINED_MODEL_NAME, 2)

    model.load_state_dict(torch.load(model_dir + '/pytorch_model.bin', map_location=torch.device('cpu')))
    
    #print("================ model loaded ===========================")
    return model.to(device)

def input_fn(serialized_input_data, request_content_type):
    #print('STARTED input_fn')
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":

        data = json.loads(serialized_input_data)
        #print("================ input sentences ===============")
        #print(data)

        
                
        tokenized_text = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=False, max_length=MAX_LEN)
        tokenized_context = tokenizer(data['context'], return_tensors='pt', padding=True, truncation=False, max_length=MAX_LEN)
        
        input_ids_text = tokenized_text['input_ids']
        attention_mask_text = tokenized_text['attention_mask']

        input_ids_context = tokenized_context['input_ids']
        attention_mask_context = tokenized_context['attention_mask']

        #print("================ tokenized text ==============")
        
        #print(tokenized_text)
        
        #print("================= tokenized context ================")
        #print(tokenized_context)

        return input_ids_text, attention_mask_text, input_ids_context, attention_mask_context

    elif request_content_type == 'text/csv':
        # Read the raw input data as CSV.
        #print('STARTED creating data_list')

        #df = pd.read_csv(StringIO(serialized_input_data), 
        #                 header=None, sep='\t')
        #print('STARTED openinig data')
        #print(serialized_input_data)
        #print(type(serialized_input_data))
        #print(serialized_input_data.split('\t'))
        
        data_list = serialized_input_data.split('\t')
        id_val = data_list[0]
        text = data_list[1]
        context = ''
        for i in data_list[2:]:
            j = i + '\n'
            context += j

        #print('STARTED encoding')

        tokenized_text = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
        tokenized_context = tokenizer(context, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
        
        input_ids_text = tokenized_text['input_ids']
        attention_mask_text = tokenized_text['attention_mask']

        input_ids_context = tokenized_context['input_ids']
        attention_mask_context = tokenized_context['attention_mask']

        #print("================ tokenized text ==============")
        
        #print(tokenized_text)
        
        #print("================= tokenized context ================")
        #print(tokenized_context)
        
        return id_val, input_ids_text, attention_mask_text, input_ids_context, attention_mask_context
    raise ValueError("Unsupported content type: {}".format(request_content_type))


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    #print('STARTED output_fn')
    #logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)



def predict_fn(input_data, model):
    #print('STARTED predict_fn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id_val, input_ids_text, attention_mask_text, input_ids_context, attention_mask_context = input_data
    input_ids_text = input_ids_text.to(device)
    attention_mask_text = attention_mask_text.to(device)
    input_ids_context = input_ids_context.to(device)
    attention_mask_context = attention_mask_context.to(device)
    #print("============== encoded data =================")
    #print(input_ids_text, attention_mask_text, input_ids_context, attention_mask_context)
    with torch.no_grad():
        y = model(input_ids_text, attention_mask_text=attention_mask_text, input_ids_context=input_ids_context, attention_mask_context=attention_mask_context)
        #print("=============== inference result =================")
        #print(y)
        probs = y.softmax(1).tolist()
        #print(probs)
    return id_val, probs

