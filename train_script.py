import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertModel
from transformers import get_linear_schedule_with_warmup
from src.dataset import WebNLGDataset
from src.utils import EuciLoss
import json
import os
import argparse




def load_model(model_path_src):
    '''
    load model
    '''
    if os.path.exists(model_path_src):
        model = BertModel.from_pretrained(training_config['model_path_src'])
    else:
        model = BertModel.from_pretrained("bert-base-uncased")

    return model



def load_trails(training_config):
    step_losses = list()
    if os.path.exists(training_config['step_losses_pth']):
        with open(training_config['step_losses_pth'], 'r') as file:
            step_losses = json.load(file)
            file.close()

    train_losses = list()
    if os.path.exists(training_config['train_losses_pth']):
        with open(training_config['train_losses_pth'], 'r') as file:
            train_losses = json.load(file)
            file.close()
    
    test_losses = list()
    if os.path.exists(training_config['test_losses_pth']):
        with open(training_config['test_losses_pth'], 'r') as file:
            test_losses = json.load(file)
            file.close()

    train_accuracy = list()
    if os.path.exists(training_config['train_accuracy_pth']):
        with open(training_config['train_accuracy_pth'], 'r') as file:
            train_accuracy = json.load(file)
            file.close()
    
    test_accuracy = list()
    if os.path.exists(training_config['test_accuracy_pth']):
        with open(training_config['test_accuracy_pth'], 'r') as file:
            test_accuracy = json.load(file)
            file.close()

    return step_losses, train_losses, test_losses, train_accuracy, test_accuracy


def train_test_loop(training_config, model, dataloader_train, dataloader_test, optimizer, creterian, step_losses, train_losses, test_losses, train_accuracy, test_accuracy, device):
    for epoch in range(training_config['num_of_epochs']):
        loss_sum_train = 0
        model.train()
        #  train loop
        for i, batch in enumerate(dataloader_train):
            batch = tuple(t.to(device) for t in batch)
            b_triple_tensor, b_triple_mask_tensor, b_lex_tensor, b_lex_mask_tensor, b_negative_lex, b_negative_lex_mask = batch

            optimizer.zero_grad()

            b_triple_vector = model(b_triple_tensor, attention_mask = b_triple_mask_tensor)[1]
            b_lex_vector = model(b_lex_tensor, attention_mask = b_lex_mask_tensor)[1]
            b_neg_lex_vector = model(b_negative_lex, attention_mask = b_negative_lex_mask)[1]

            loss = creterian(b_triple_mask_tensor, b_lex_vector, b_neg_lex_vector)

            loss.backward()
            optimizer.step()
            loss_scalar = loss.item()
            loss_sum_train += loss_scalar
            step_losses.append(loss_scalar)

            #  b_predicts = torch.argmax(b_outputs, dim=-1)
            #  correct += (b_predicts == b_labels).sum().item()

        train_loss = loss_sum_train / len(dataloader_train)
        train_losses.append(train_loss)




        loss_sum_test = 0
        correct = 0

        model.eval() 
        #  test_loop
        for i, batch in enumerate(dataloader_test):
            batch = tuple(t.to(device) for t in batch)
            b_triple_tensor, b_triple_mask_tensor, b_lex_tensor, b_lex_mask_tensor, b_negative_lex, b_negative_lex_mask = batch

            with torch.no_grad():
                b_triple_vector = model(b_triple_tensor, attention_mask = b_triple_mask_tensor)[1]
                b_lex_vector = model(b_lex_tensor, attention_mask = b_lex_mask_tensor)[1]
                b_neg_lex_vector = model(b_negative_lex, attention_mask = b_negative_lex_mask)[1]

                loss = creterian(b_triple_mask_tensor, b_lex_vector, b_neg_lex_vector)

                loss_scalar = loss.item()
                loss_sum_test += loss_scalar

        test_loss = loss_sum_test / len(dataloader_test)
        test_losses.append(test_loss)



        print(f'Epoch: {epoch+1} \n Train Loss: {train_loss:.6f}, train Test Loss: {test_loss:.6f}')



def save_trails(training_config, step_losses, train_losses, test_losses, train_accuracy, test_accuracy):
    with open(training_config['step_losses_pth'], 'w') as file:
        json.dump(step_losses, file)
        file.close()

    with open(training_config['train_losses_pth'], 'w') as file:
        json.dump(train_losses, file)
        file.close()
    
    with open(training_config['test_losses_pth'], 'w') as file:
        json.dump(test_losses, file)
        file.close()

    with open(training_config['train_accuracy_pth'], 'w') as file:
        json.dump(train_accuracy, file)
        file.close()
    
    with open(training_config['test_accuracy_pth'], 'w') as file:
        json.dump(test_accuracy, file)
        file.close()


def train(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    '''
    model
    load model and the history
    '''
    model = load_model(training_config['model_path_src']).to(device)


    #  load the losses history
    step_losses, train_losses, test_losses, train_accuracy, test_accuracy = load_trails(training_config)



    '''
    dataloader
    '''
    train_dataset = WebNLGDataset(data_path = './data/en/train', max_length = 128)
    test_dataset = WebNLGDataset(data_path = './data/en/dev', max_length = 128)

    dataloader_train = DataLoader(train_dataset, batch_size = training_config['batch_size'], shuffle = True)
    dataloader_test  = DataLoader(test_dataset, batch_size  = training_config['batch_size'], shuffle = False)


    '''
    optimizer
    '''
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    # Separate the `weight` parameters from the `bias` parameters.
    # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
    # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
    optimizer_grouped_parameters = [
        # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.1},

        # Filter for parameters which *do* include those.
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    # Note - `optimizer_grouped_parameters` only includes the parameter values, not
    # the names.

    optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr = training_config['learning_rate'],
            eps = 1e-8
            )

    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = 0, # Default value in run_glue.py
            num_training_steps = len(dataloader_train) * training_config['num_of_epochs']
            )




    '''
    creterian
    '''
    creterian = EuciLoss().to(device)


    '''
    train  and validate loops
    '''
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    train_test_loop(training_config, model, dataloader_train, dataloader_test, optimizer, creterian, step_losses, train_losses, test_losses, train_accuracy, test_accuracy, device)


        
    '''    
    save model and data
    '''

    model = model.to('cpu').module
    torch.save(model.state_dict(), training_config['model_path_dst'])

    #  save the loss of the steps
    save_trails(training_config, step_losses, train_losses, test_losses, train_accuracy, test_accuracy)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs"    , type=int   , help="number of epochs"                                  , default=20)
    parser.add_argument("--batch_size"       , type=int   , help="batch size"                                        , default=512)
    parser.add_argument("--learning_rate"    , type=float , help="learning rate"                                     , default=5e-4)
    parser.add_argument("--weight_decay"     , type=float , help="weight_decay"                                      , default=1e-4)
    parser.add_argument("--vocab_size"       , type=int   , help="vocab size"                                        , default=21128)
    parser.add_argument("--embedding_dim"    , type=int   , help="embedding dimmention"                              , default=512)
    parser.add_argument("--num_labels"       , type=int   , help="types of labels"                                   , default=6)
    parser.add_argument("--sequence_length"  , type=int   , help="sequence_length"                                   , default=128)
    parser.add_argument("--model_path_dst"   , type=str   , help="the directory to save model"                       , default='./saved_models/saved_dict.pth')
    parser.add_argument("--model_path_src"   , type=str   , help="the directory to load model"                       , default='./saved_models/saved_dict.pth')
    parser.add_argument("--step_losses_pth"  , type=str   , help="the path of the json file that saves step losses"  , default='./trails/step_losses.json')
    parser.add_argument("--train_losses_pth" , type=str   , help="the path of the json file that saves train losses" , default='./trails/train_losses.json')
    parser.add_argument("--test_losses_pth"  , type=str   , help="the path of the json file that saves test losses"  , default='./trails/test_losses.json')
    parser.add_argument("--train_accuracy_pth" , type=str   , help="the path of the json file that saves train accuracy" , default='./trails/train_accuracy.json')
    parser.add_argument("--test_accuracy_pth"  , type=str   , help="the path of the json file that saves test accuracy"  , default='./trails/test_accuracy.json')

    
    args = parser.parse_args()

    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)

    train(training_config)
