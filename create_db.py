import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
from transformers import BertModel
import sqlite3
import json
import copy
import re
import random
from src.benchmark_reader import Benchmark, select_files
import os



def load_model(model_path_src):
    '''
    load model
    '''
    if os.path.exists(model_path_src):
        model = BertModel.from_pretrained(model_path_src)
    else:
        model = BertModel.from_pretrained("bert-base-uncased")

    return model



if __name__ == "__main__":

    database_path = './vecBase.db'
    model_path_src = './saved_models/'
    max_length = 64
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    ''' 
    import model
    '''
    model = load_model(model_path_src).to(device)



    '''
    connect to db
    '''
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    CREATE_VEC_TABLE = "CREATE TABLE IF NOT EXISTS vecTable (vector TEXT, target TEXT)"
    cursor.execute(CREATE_VEC_TABLE)


    '''
    read and process the datas
    '''
    b = Benchmark()
    files = select_files('data/en/train/')

    files = files[0: 16]

    b.fill_benchmark(files)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


    for i in range(len(b.entries)):
        entry = b.entries[i]
        for j in range(len(entry.modifiedtripleset.triples)):
            triple = list()
            triple.append(entry.modifiedtripleset.triples[j].s)
            triple.append(entry.modifiedtripleset.triples[j].p)
            triple.append(entry.modifiedtripleset.triples[j].o)

            triple_str = ' '.join(triple)


            for k in range(len(entry.lexs)):
                lex = entry.lexs[k].lex
                encoding = tokenizer(lex, padding = 'max_length', truncation=True, max_length=max_length)
                encoded_lex = encoding['input_ids']
                lex_mask    = encoding['attention_mask']

                lex_tensor         = torch.tensor([encoded_lex] , dtype = torch.int32).to(device)
                lex_mask_tensor    = torch.tensor([lex_mask]      , dtype = torch.int32).to(device)

                with torch.no_grad():
                    lex_vector = model(lex_tensor, attention_mask = lex_mask_tensor)[1].squeeze(0).to('cpu')
                    cursor.execute(f"INSERT INTO vecTable (vector, target) VALUES (?,?)", (json.dumps(lex_vector.tolist()), lex))


    connection.commit()
    connection.close()
