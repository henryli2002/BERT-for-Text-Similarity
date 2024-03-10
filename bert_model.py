from transformers import BertTokenizer, BertForTokenClassification
import torch 
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',
    cache_dir=None,
    force_download=False,
)

df_train = pd.read_csv('Chinese-STS-B/sts-b-train.txt', sep='\t').values
df_dev = pd.read_csv('Chinese-STS-B/sts-b-dev.txt', sep='\t').values
df_test = pd.read_csv('Chinese-STS-B/sts-b-test.txt', sep='\t').values

def tokenizer(line):
    data = tokenizer.encode(
        text=df_train[line][0],
        text_pair=df_train[line][1],
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        max_length=30,  
        return_tensors=None,
    )   
    return data
