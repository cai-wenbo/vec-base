from numpy import vectorize
import numpy as np
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


class neighbourExtractor():
    def __init__(self, model_path_src, vector_list):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = load_model(model_path_src)
        self.vector_list = vector_list

    def __call__(self, input_text, n):
        encoding = self.tokenizer(input_text, padding = 'max_length', truncation=True, max_length=max_length)
        encoded_input = encoding['input_ids']
        input_mask    = encoding['attention_mask']

        input_tensor         = torch.tensor(encoded_input , dtype = torch.int32)
        input_mask_tensor    = torch.tensor(input_mask      , dtype = torch.int32)


        with torch.no_grad():
            input_vector = model(input_tensor, attention_mask = input_mask_tensor)[1].to_list()

        scores = list()
        for i in range(len(self.vector_list)):
            scores.append(np.sum(np.multiply(np.array(input_vector), np.array(self.vector_list[i]))))
        indexed_scores = list(enumerate(scores))
        sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        top_n_neighbours = [index for index, _ in sorted_scores[:n]]
        return top_n_neighbours




if __name__ == "__main__":

    database_path = './vecBase.db'
    model_path_src = './saved_models/'
    max_length = 64
    n = 5
    


    '''
    connect to db
    '''
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()



    '''
    read db
    '''
    cursor.execute(f"SELECT vector FROM vecTable")
    vector_json_list = cursor.fetchall()
    
    vector_list = list()
    for i in range(len(vector_json_list)):
        vector_list.append(json.loads(vector_list[i]))
    


    '''
    extract
    '''
    while True:
        query = input("enter the query:\n")
        neighbours_extractor = neighbourExtractor(model_path_src, vector_list)
        result = neighbours_extractor(n, query)

        for i in range(len(result)):
            cursor.execute(f"SELECT result FROM vecTable LIMIT 1 OFFSET {i}")
            text = cursor.fetchone()
            print(text)