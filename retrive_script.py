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

        input_tensor         = torch.tensor([encoded_input] , dtype = torch.int32)
        input_mask_tensor    = torch.tensor([input_mask]      , dtype = torch.int32)


        with torch.no_grad():
            input_vector = self.model(input_tensor, attention_mask = input_mask_tensor)[1].squeeze(0).tolist()


        scores = list()
        for i in range(len(self.vector_list)):
            input_array = np.array(input_vector)
            record_array = np.array(self.vector_list[i])
            #  scores.append(np.sum(np.multiply(input_array,record_array)))
            scores.append(np.dot(input_array, record_array) / (np.linalg.norm(input_array) * np.linalg.norm(record_array)))
            #  scores.append(-np.linalg.norm(input_array - record_array))
            print(scores[i])

        indexed_scores = list(enumerate(scores))
        sorted_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        top_n_neighbours = [index for index, _ in sorted_scores[:n]]
        return top_n_neighbours




if __name__ == "__main__":

    database_path = './vecBase.db'
    model_path_src = './saved_models/'
    max_length = 64
    n = 20
    


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
        vector_json, = vector_json_list[i]
        vector_list.append(json.loads(vector_json))

    


    '''
    extract
    '''
    query = input("enter the query:\n")
    while query is not None:
        neighbours_extractor = neighbourExtractor(model_path_src, vector_list)
        result = neighbours_extractor(query, n)
        print(result)
        for i in result:
            cursor.execute(f"SELECT target FROM vecTable LIMIT 1 OFFSET {i}")
            text = cursor.fetchone()
            print(text)
        query = input("enter the query:\n")
