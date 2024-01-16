from numpy import tri
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
import copy
import re
import random
from .benchmark_reader import Benchmark, select_files


class tripleLexPair():
    def __init__(self, encoded_triple, triple_mask, encoded_lex, lex_mask):
        self.encoded_triple = encoded_triple
        self.triple_mask    = triple_mask
        self.encoded_lex    = encoded_lex
        self.lex_mask       = lex_mask


class WebNLGDataset(Dataset):
    '''
    read the WebNLG dataset
    and return the (triple, text) data for training
    '''
    def __init__(self, data_path, max_length, num_of_negative = 4, selected_field = None):
        '''
        '''
        b = Benchmark()
        files = select_files(data_path)

        if selected_field is not None:
            files = files[selected_field[0]: selected_field[1]]

        b.fill_benchmark(files)


        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        tripleLexPair_list = list() 

        for i in range(len(b.entries)):
            entry = b.entries[i]
            for j in range(len(entry.modifiedtripleset.triples)):
                triple = list()
                triple.append(entry.modifiedtripleset.triples[j].s)
                triple.append(entry.modifiedtripleset.triples[j].p)
                triple.append(entry.modifiedtripleset.triples[j].o)

                triple_str = ' '.join(triple)

                encoding = tokenizer(triple_str, padding = 'max_length', truncation=True, max_length=max_length)
                encoded_triple = encoding['input_ids']
                triple_mask    = encoding['attention_mask']

                for k in range(len(entry.lexs)):
                    lex = entry.lexs[k].lex

                    encoding = tokenizer(lex, padding = 'max_length', truncation=True, max_length=max_length)
                    encoded_lex = encoding['input_ids']
                    lex_mask    = encoding['attention_mask']

                    
                    tripleLexPair_list.append(tripleLexPair(encoded_triple, triple_mask, encoded_lex, lex_mask))


        self.num_of_negative = num_of_negative
        self.tripleLexPair_list = tripleLexPair_list

    def __len__(self):
        return len(self.tripleLexPair_list)


    def __getitem__(self, idx):
        
        random_idx = random.randint(0, len(self.tripleLexPair_list) -1)


        negative_lex = self.tripleLexPair_list[random_idx].encoded_lex
        negative_lex_mask = self.tripleLexPair_list[random_idx].lex_mask



        #  tensorlize
        triple_tensor      = torch.tensor(self.tripleLexPair_list[idx].encoded_triple , dtype = torch.int32)
        triple_mask_tensor = torch.tensor(self.tripleLexPair_list[idx].triple_mask    , dtype = torch.int32)
        lex_tensor         = torch.tensor(self.tripleLexPair_list[idx].encoded_lex    , dtype = torch.int32)
        lex_mask_tensor    = torch.tensor(self.tripleLexPair_list[idx].lex_mask       , dtype = torch.int32)

        negative_lex_tensor = torch.tensor(negative_lex, dtype = torch.int32)
        negative_lex_mask_tensor = torch.tensor(negative_lex_mask, dtype = torch.int32)

        

        return triple_tensor, triple_mask_tensor, lex_tensor, lex_mask_tensor, negative_lex_tensor, negative_lex_mask_tensor
