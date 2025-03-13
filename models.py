
import math
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, AutoConfig, PreTrainedModel


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


################################################################################################
# Abstract base class for models
################################################################################################

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        # Every model class should implement these so they can do saving/loading
        self.save_dir = None
        self.name = None

        # This does not have to be overridden in every model; just in model
        # classes that act as wrappers for HuggingFace models
        self.model = None

    def forward(self, batch):

        # Input: batch - a dictionary containing any inputs that the model needs
        # Output: another dictionary, containing any outputs that will be needed from the model
        raise NotImplementedError

    def trainable_parameters(self):

        # Yield the model's trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param

    def save(self):
        logging.info("Saving model checkpoint to %s", self.save_dir)
        save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        if not isinstance(self.model, PreTrainedModel):
            state_dict = self.state_dict()
            torch.save(state_dict, os.path.join(self.save_dir, self.name + ".weights"))
        else:
            output_dir = os.path.join(self.save_dir, self.name)
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)

    def load(self, model_name=None):
        
        # Default to loading the best saved weights for this model
        # (if model_name is provided, this default is overridden to
        # instead load a different pretrained model)
        if model_name is None:
            model_name = os.path.join(self.save_dir, self.name)

        logging.info("Loading model checkpoint from %s", model_name)

        if isinstance(self.model, GPT2LMHeadModel):
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        else:
            if torch.cuda.is_available():
                self.load_state_dict(torch.load(model_name + ".weights"))
            else:
                self.load_state_dict(torch.load(model_name + ".weights", map_location=torch.device('cpu')))





################################################################################################
# Classifier
################################################################################################

# Input: 8 - zero vs one as 2 separate indices
class MLPClassifierRationalRules(Model):

    def __init__(self, n_features=4, hidden_size=None, n_layers=None, dropout=0.5, nonlinearity="ReLU", save_dir=None, model_name=None):
        super(MLPClassifierRationalRules, self).__init__()

        self.save_dir = save_dir
        self.name = model_name

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.nonlinearity = nonlinearity

        self.drop = nn.Dropout(dropout)

        self.in_layer = nn.Linear(self.n_features*2, self.hidden_size)
        self.out_layer = nn.Linear(self.hidden_size, 1)

        for layer_index in range(self.n_layers):
            setattr(self, "layer" + str(layer_index), nn.Linear(self.hidden_size, self.hidden_size))

        if self.nonlinearity == "ReLU":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "sigmoid":
            self.nonlinearity = nn.Sigmoid()

        # For squeezing the output into [0,1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, reduction="mean"):

        hidden = self.in_layer(batch["input_ids"])
        for layer_index in range(self.n_layers):
            hidden = getattr(self, "layer" + str(layer_index))(hidden)
            if layer_index % 2 == 0 and layer_index:
                hidden += skip
            hidden = self.nonlinearity(hidden)
            hidden = self.drop(hidden)
            if layer_index % 2 == 0:
                skip = hidden
        output = self.out_layer(hidden)
        probs = self.sigmoid(output)

        loss = None
        acc = None
        if "labels" in batch:
            loss_fct = nn.BCELoss(reduction=reduction)
            loss = loss_fct(probs, batch["labels"])

            # Compute accuracy
            preds = (probs > 0.5)
            correct = torch.sum(preds == batch["labels"])
            incorrect = torch.sum(preds != batch["labels"])
            acc = correct * 1.0 / (correct + incorrect)
            acc = acc.item()
            correct_array = (preds == batch['labels'])


        return {"probs" : probs, "loss" : loss, "accuracy" : acc, "correct_array": probs}


class MLPClassifier(Model):

    def __init__(self, n_features=4, hidden_size=None, n_layers=None, dropout=0.5, nonlinearity="ReLU", save_dir=None, model_name=None):
        super(MLPClassifier, self).__init__()

        self.save_dir = save_dir
        self.name = model_name

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.nonlinearity = nonlinearity

        self.drop = nn.Dropout(dropout)

        self.in_layer = nn.Linear(self.n_features*15, self.hidden_size)
        self.out_layer = nn.Linear(self.hidden_size, 10)

        for layer_index in range(self.n_layers):
            setattr(self, "layer" + str(layer_index), nn.Linear(self.hidden_size, self.hidden_size))

        if self.nonlinearity == "ReLU":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "sigmoid":
            self.nonlinearity = nn.Sigmoid()

        # For squeezing the output into [0,1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch, reduction="mean"):

        hidden = self.in_layer(batch["input_ids"])
        for layer_index in range(self.n_layers):
            hidden = getattr(self, "layer" + str(layer_index))(hidden)
            if layer_index % 2 == 0 and layer_index:
                hidden += skip
            hidden = self.nonlinearity(hidden)
            hidden = self.drop(hidden)
            if layer_index % 2 == 0:
                skip = hidden
        output = self.out_layer(hidden)
        output = output.reshape((-1, 2))

        loss = None
        acc = None
        if "labels" in batch:
            labels = batch["labels"].view(-1)
            loss_fct = nn.CrossEntropyLoss(reduction=reduction, ignore_index=2)
            loss = loss_fct(output, labels)
            # print('output shape', output.shape)

            probs = nn.Softmax(dim=1)(output)
            # print(probs)
            probs = probs[:, 1]

            indices = torch.where(labels != 2)
            probs = probs[indices]
            labels = labels[indices]

            # Compute accuracy
            preds = (probs > 0.5)
            correct = torch.sum(preds == labels)
            incorrect = torch.sum(preds != labels)
            acc = correct * 1.0 / (correct + incorrect)
            acc = acc.item()
            correct_array = (preds == labels)
            # print('correct array from model pass', correct_array)

            # if acc != 1:
            #     print('wrong example', batch['input_ids'])
            #     print('preds', preds)
            #     print('true lables', labels)
        return {"probs" : probs, "loss" : loss, "accuracy" : acc, "correct_array": correct_array}


class Transformer(Model):

    def __init__(self, n_features=4, hidden_size=None, n_layers=None, dropout=0.5, nonlinearity="ReLU", save_dir=None, model_name=None):
        super(Transformer, self).__init__()

        self.save_dir = save_dir
        self.name = model_name

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.nonlinearity = nonlinearity

        self.drop = nn.Dropout(dropout)

        self.in_layer = nn.Linear(self.n_features*3, self.hidden_size)
        
        # self.positional_encoding = PositionalEncoding(hidden_size)
        self.num_heads = 8
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=self.num_heads, batch_first=True), num_layers=n_layers)
        self.out_layer = nn.Linear(self.hidden_size, 2)

    def forward(self, batch, reduction="mean"):

        # print("batch in", batch['input_ids'])
        input_ids = batch["input_ids"].reshape(-1, 5, 9)
        # print(input_ids)
        hidden = self.in_layer(input_ids)
        # print('hidden', hidden.shape)
        
        # print("labels 2", (batch['labels'] == 2).shape)
        src_key_padding_mask = (batch['labels'] == 2)
        output = self.transformer(hidden, src_key_padding_mask=src_key_padding_mask)
        # print(output.shape)

        output = self.out_layer(output)
        # print("before rehape", output, output.shape)
        output = output.reshape((-1, 2))
        # print("after rehape", output, output.shape)

        loss = None
        acc = None
        if "labels" in batch:
            labels = batch["labels"].view(-1)
            # print('LABELS', labels, labels.shape)
            loss_fct = nn.CrossEntropyLoss(reduction=reduction, ignore_index=2)
            loss = loss_fct(output, labels)
            # print('output shape', output.shape)

            probs = nn.Softmax(dim=1)(output)
            # print(probs)
            probs = probs[:, 1]

            indices = torch.where(labels != 2)
            probs = probs[indices]
            labels = labels[indices]

            # Compute accuracy
            preds = (probs > 0.5)
            correct = torch.sum(preds == labels)
            incorrect = torch.sum(preds != labels)
            acc = correct * 1.0 / (correct + incorrect)
            acc = acc.item()
            correct_array = (preds == labels)
            # print('correct array from model pass', correct_array)

            # if acc != 1:
            #     print('wrong example', batch['input_ids'])
            #     print('preds', preds)
            #     print('true lables', labels)
        return {"probs" : probs, "loss" : loss, "accuracy" : acc, "correct_array": correct_array}

class LSTM(Model):

    def __init__(self, n_features=4, hidden_size=None, n_layers=None, dropout=0.5, nonlinearity="ReLU", save_dir=None, model_name=None):
        super(LSTM, self).__init__()

        self.save_dir = save_dir
        self.name = model_name

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.nonlinearity = nonlinearity

        self.drop = nn.Dropout(dropout)

        self.in_layer = nn.Linear(self.n_features*3, self.hidden_size)
        
        # self.positional_encoding = PositionalEncoding(hidden_size)
        self.num_heads = 8
        self.LSTM = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.out_layer = nn.Linear(2 * self.hidden_size, 2)

    def strip_two(self, labels):
        # labels batch x 5
        len_seq = torch.sum(labels != 2, dim=1)
        max_len = max(len_seq)
        return labels[:, :max_len]

    def forward(self, batch, reduction="mean"):

        # print("batch in", batch['input_ids'])
        input_ids = batch["input_ids"].reshape(-1, 5, 9)
        # print(input_ids)
        input_feat = self.in_layer(input_ids)
        # print('input feat', input_feat.shape)
        
        # print("labels 2", (batch['labels'] == 2).shape)
        # print('batch labels', batch['labels'] != 2)
        len_seq = torch.sum(batch['labels'] != 2, dim=1)
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(input_feat, len_seq, batch_first=True, enforce_sorted=False)

        # print(packed_seq)
        # print(torch.zeros_like(input_feat).shape)
        output, _ = self.LSTM(packed_seq, (torch.zeros(2 * self.n_layers, input_feat.shape[0], self.hidden_size), \
                                        torch.zeros(2 * self.n_layers, input_feat.shape[0], self.hidden_size)))
        # print(output.shape)
        output =  torch.nn.utils.rnn.unpack_sequence(output)
        output = torch.nn.utils.rnn.pad_sequence(output, batch_first=True)
        # print("after pad", output.shape)
        output = self.out_layer(output)
        # print("before rehape", output, output.shape)
        output = output.reshape((-1, 2))
        # print("after rehape", output, output.shape)

        loss = None
        acc = None
        if "labels" in batch:
            labels_stripped = self.strip_two(batch["labels"])
            labels = labels_stripped.reshape(-1)
            # print('LABELS', labels, labels.shape)
            loss_fct = nn.CrossEntropyLoss(reduction=reduction, ignore_index=2)
            loss = loss_fct(output, labels)
            # print('output shape', output.shape)

            probs = nn.Softmax(dim=1)(output)
            # print(probs)
            probs = probs[:, 1]

            indices = torch.where(labels != 2)
            probs = probs[indices]
            labels = labels[indices]

            # Compute accuracy
            preds = (probs > 0.5)
            correct = torch.sum(preds == labels)
            incorrect = torch.sum(preds != labels)
            acc = correct * 1.0 / (correct + incorrect)
            acc = acc.item()
            correct_array = (preds == labels)
            # print('correct array from model pass', correct_array)

            # if acc != 1:
            #     print('wrong example', batch['input_ids'])
            #     print('preds', preds)
            #     print('true lables', labels)
        return {"probs" : probs, "loss" : loss, "accuracy" : acc, "correct_array": correct_array}

if __name__ == "__main__":
    # inputs = [[0,0,0,0], [0,1,1,0], [0,1,1,0], [1,1,1,0], [1,1,1,0]]
    # labels = [0,0,0,1,1]

    # batch = {"input_ids" : torch.FloatTensor(inputs), "labels" : torch.FloatTensor(labels).unsqueeze(1)}
    # model = MLPClassifier(n_features=4, hidden_size=7, n_layers=3, dropout=0.0, nonlinearity="ReLU")
    # print(model(batch))
    inputs= torch.tensor([[1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0.,
        0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0.,
        0., 1., 0., 1., 0., 0., 0., 1., 0.], [1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0.,
        0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0.,
        0., 1., 0., 1., 0., 0., 0., 1., 0.]])
    # [[0,0,0,0,1,0,0,0,0], 
    #           [0,1,0,0,1,0,0,1,0], 
    #           [0,1,0,0,0,0,0,1,0], 
    #           [1,0,0,0,1,0,0,0,1], 
    #           [0,0,1,0,0,0,1,0,0]]
    labels = [[0,1,1,1,2], [0,1,1,2,2]]
    
    batch = {"input_ids" : torch.FloatTensor(inputs), "labels" : torch.LongTensor(labels)}
    print(batch['input_ids'].shape)
    print(batch['labels'].shape)
    # model = MLPClassifier(n_features=4, hidden_size=7, n_layers=3, dropout=0.0, nonlinearity="ReLU")
    model = LSTM(n_features=3, hidden_size=128, n_layers=5, dropout=0.1, nonlinearity="ReLU")
    print(model(batch))







