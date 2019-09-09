import os
import csv
import shutil
import zipfile
import gzip
import pickle
import itertools
import urllib.parse
import urllib.request
from collections import Counter
import functools

import numpy as np
import torch
import torch.utils.data
from nltk import word_tokenize


from flask import Flask, render_template, request, jsonify, Response


### CODE BELOW IS COPIED FROM THE NOTEBOOK ###

class Vocab(object):
    """Vocabulary class to provide token to id correpondance"""
    END_TOKEN = '<end>'
    START_TOKEN = '<start>'
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    def __init__(self, special_tokens=None):
        """
        Initialize the vocabulary class

        :param special_tokens: Default value = None) A list of special tokens. The PAD token should be the first in the list, if used.

        """
        super().__init__()

        self.special_tokens = special_tokens

        self.token2id = {}
        self.id2token = {}

        self.token_counts = Counter()

        if self.special_tokens is not None:
            self.add_document(self.special_tokens)

    def add_document(self, document, rebuild=True):
        """
        Process the document and add tokens from the it to the vocabulary

        :param document: A list of tokens in the document
        :param rebuild: Default value = True) Whether to rebuild the token2id correspondance or not

        """
        for token in document:
            self.token_counts[token] += 1

            if token not in self.token2id:
                self.token2id[token] = len(self.token2id)

        if rebuild:
            self._rebuild_id2token()

    def add_documents(self, documents):
        """
        Process a list of documents and tokens from the them to the vocabulary

        :param documents: A list of documents, where each document is a list of tokens

        """
        for doc in documents:
            self.add_document(doc, rebuild=False)

        self._rebuild_id2token()

    def _rebuild_id2token(self):
        """Revuild the token to id correspondance"""
        self.id2token = {i: t for t, i in self.token2id.items()}

    def get(self, item, default=None):
        """
        Given a token, return the corresponding id

        :param item: A token
        :param default: Default value = None) Default value to return if token is not present in the vocabulary

        """
        return self.token2id.get(item, default)

    def __getitem__(self, item):
        """
        Given a token, return the corresponding id

        :param item: A token

        """
        return self.token2id[item]

    def __contains__(self, item):
        """
        Check if a token is present in the vocabulary

        :param item: A token

        """
        return item in self.token2id

    def __len__(self):
        """ """
        return len(self.token2id)

    def __str__(self):
        """Get a string representation of the vocabulary"""
        return f'{len(self)} tokens'

    def save(self, filename):
        """
        Save the vocabulary to a csv file. See the `load` method.

        :param filename: Path the file

        """
        with open(filename, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['token', 'counts', 'is_special'])
            writer.writeheader()
            for idx in range(len(self.token2id)):
                token = self.id2token[idx]
                is_special = 1 if token in self.special_tokens else 0
                writer.writerow({'token': token, 'counts': self.token_counts[token], 'is_special': is_special})

    @staticmethod
    def load(filename):
        """
        Load the vocabulary from a csv file. See the `save` method.

        :param filename: 

        """
        with open(filename, 'r') as csv_file:
            token2id = {}
            tokens_counts = {}
            special_tokens = []
            reader = csv.DictReader(csv_file)
            for i, row in enumerate(reader):
                token2id[row['token']] = i
                tokens_counts[row['token']] = int(row['counts'])
                if bool(int(row['is_special'])):
                    special_tokens.append(row['token'])

        vocab = Vocab()
        vocab.token2id = token2id
        vocab.token_counts = Counter(tokens_counts)
        vocab.special_tokens = special_tokens
        vocab._rebuild_id2token()

        return vocab


class SubtitlesDialogDataset(torch.utils.data.Dataset):
    """ A conversational dialog dataset with query-response pairs  """
    def __init__(self, filename, vocab=None, max_lines = 1000, max_len=50, max_vocab_size=50000):
        """
        Initialize a conversational dialog dataset with query-response pairs        

        :param filename: Path to the OpenSubstitles dataset
        :param vocab:  (Default value = None) Vocabulary, will be created if None
        :param max_lines:  (Default value = 1000) Limit the number of lines to read from the dataset file
        :param max_len:  (Default value = 50) Maximum length of the sentences
        :param max_vocab_size:  (Default value = 50000) Maximum size of the vocabulary

        """

        self.lines = []
        with gzip.open(filename, 'rb') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break

                tokens = word_tokenize(line.decode('utf-8'))
                self.lines.append(tokens)

        self.max_lines = min(len(self.lines), max_lines)
                
        if vocab is None:
            vocab = Vocab(special_tokens=[Vocab.PAD_TOKEN, Vocab.START_TOKEN, Vocab.END_TOKEN, Vocab.UNK_TOKEN])
            vocab.add_documents(self.lines)
            vocab.prune_vocab(max_vocab_size)

            print(f'Created vocab: {vocab}')

            
        if max_len is None:
            max_len = max(len(s) for s in itertools.chain.from_iterable(self.sentences))
            print(f'Calculed max len: {max_len}')
        
        self.vocab = vocab
        self.max_len = max_len
        
    def _pad_sentnece(self, sent):
        """
        Cut the sentence if needed and pad it to the maximum len

        :param sent: The input sentnece

        """
        sent = sent[:self.max_len - 1] + [Vocab.END_TOKEN,]
        
        nb_pad = self.max_len - len(sent)
        sent = sent + [Vocab.PAD_TOKEN,] * nb_pad
        
        return sent
        
    def _process_sent(self, sent):
        """
        Cut, pad, and convert the sentence from tokens to indices using the vocabulary

        :param sent: The input sentence

        """
        sent = self._pad_sentnece(sent)
        sent = [self.vocab[t] if t in self.vocab else self.vocab[Vocab.UNK_TOKEN] for t in sent]
        
        sent = np.array(sent, dtype=np.long)
        return sent
        
    def __getitem__(self, index):
        """
        Create a pair of query-reponse using two consequtive lines in the dataset and return it

        :param index: Index of the query line. The reponse is the next line.

        """
        query = self.lines[index]
        response = self.lines[index+1]
        
        query = self._process_sent(query)
        response = self._process_sent(response)        
        
        return query, response
    
    def __len__(self):
        """ Return the total length of the dataset """
        return self.max_lines - 1

def softmax_masked(inputs, mask, dim=1, epsilon=0.000001):
    """
    Perform the softmas operation on a batch of masked sequences of different lengths

    :param inputs: Input sequences, a 2d array of the shape (batch_size, max_seq_len)
    :param mask: Mask, an array of 1 and 0
    :param dim:  (Default value = 1) Dimension of the softmax operation
    :param epsilon:  (Default value = 0.000001)

    """
    inputs_exp = torch.exp(inputs)
    inputs_exp = inputs_exp * mask.float()
    inputs_exp_sum = inputs_exp.sum(dim=dim)
    inputs_attention = inputs_exp / (inputs_exp_sum.unsqueeze(dim) + epsilon)

    return inputs_attention

class Seq2SeqAttentionModel(torch.nn.Module):
    """ A more advanced GRU-based sequence-to-sequence model with attention """
    def __init__(self, vocab_size, embedding_size, hidden_size, teacher_forcing,
                 max_len,trainable_embeddings, start_index, end_index, pad_index, W_emb=None):
        """
        Initialize the model

        :param vocab_size: The size of the vocabulary
        :param embedding_size: Dimension of the embeddings
        :param hidden_size: The size of the hidden layers, including GRU
        :param teacher_forcing: The probability of teacher forcing
        :param max_len: Maximum length of the sequences
        :param trainable_embeddings: Whether the embedding layer will be trainable or frozen
        :param start_index: Index of the START token in the vocabulary
        :param end_index: Index of the END token in the vocabulary
        :param pad_index: Index of the PAD token in the vocabulary
        :param W_emb:  (Default value = None) Initial values of the embedding layer, a numpy array

        """

        super().__init__()

        self.teacher_forcing = teacher_forcing
        self.max_len = max_len
        self.start_index = start_index
        self.end_index = end_index
        self.pad_index = pad_index
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=pad_index)
        if W_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(W_emb))
        if not trainable_embeddings:
            self.embedding.weight.requires_grad = False

        self.encoder = torch.nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.decoder = torch.nn.GRUCell(embedding_size, hidden_size)

        self.attention_decoder = torch.nn.Linear(hidden_size, hidden_size)
        self.attention_encoder = torch.nn.Linear(hidden_size, hidden_size)        
        self.attention_reduce = torch.nn.Linear(hidden_size, 1, bias=False)                
        self.decoder_hidden_combine = torch.nn.Linear(hidden_size * 2, hidden_size)
        
        self.decoder_projection = torch.nn.Linear(hidden_size, vocab_size)

            
    def encode(self, inputs):
        """
        Encode input sentence and return the all hidden states and the input mask

        :param inputs: The input sentence

        """
        batch_size = inputs.size(0)
        inputs_mask = (inputs != self.pad_index).long()
        inputs_lengths = torch.sum(inputs_mask, dim=1)
        
        inputs_emb = self.embedding(inputs)
        outputs, h = self.encoder(inputs_emb)
        
        return outputs, inputs_mask
    
    def decode(self, encoder_hiddens, inputs_mask, targets=None):
        """
        Decode the response given the all hidden states of the encoder

        :param encoder_hiddens: Hidden states of the decoder
        :param inputs_mask: Input mask
        :param targets:  (Default value = None) True decoding targets to be used for teacher forcing

        """
        batch_size = encoder_hiddens.size(0)

        outputs_logits = []
        decoder_hidden = torch.zeros_like(encoder_hiddens[:,0,:])
        decoder_inputs = torch.full_like(decoder_hidden[:, 0], self.start_index).long()
        for i in range(self.max_len):
            decoder_inputs_emb = self.embedding(decoder_inputs)
            
            att_enc = self.attention_encoder(encoder_hiddens)
            att_dec = self.attention_decoder(decoder_hidden)
            att = torch.tanh(att_enc + att_dec.unsqueeze(1))
            att_reduced = self.attention_reduce(att).squeeze(-1)
            att_normazlied = softmax_masked(att_reduced, inputs_mask)

            decoder_hidden_att = torch.sum(encoder_hiddens * att_normazlied.unsqueeze(-1), dim=1)
            decoder_hidden_combined = self.decoder_hidden_combine(torch.cat([decoder_hidden, decoder_hidden_att], dim=-1))
            
            decoder_hidden = self.decoder(decoder_inputs_emb, decoder_hidden_combined)
            
            decoder_output_logit = self.decoder_projection(decoder_hidden)
            
            if np.random.rand() < self.teacher_forcing and targets is not None:
                decoder_inputs = targets[:, i]
            else:
                decoder_inputs = decoder_output_logit.argmax(dim=1).long()
            
            outputs_logits.append(decoder_output_logit)
            
        outputs_logits = torch.stack(outputs_logits, dim=1)
            
        return outputs_logits
        
    def forward(self, inputs, targets=None):
        """
        Encode the input query and decode the response

        :param inputs: The input sentence
        :param targets:  (Default value = None) True decoding targets

        """
        encoder_hiddens, inputs_mask = self.encode(inputs)
        outputs_logits = self.decode(encoder_hiddens, inputs_mask, targets)

        return outputs_logits

def load_model(model_class, filename):
    """
    Create the model of the given class and load the checkpoint from the given file

    :param model_class: Model class
    :param filename: Path to the checkpoint

    """
    def _map_location(storage, loc):
        """ A utility function to load a trained on a GPU model to the CPU """
        return storage

    # load trained on GPU models to CPU
    map_location = None
    if not torch.cuda.is_available():
        map_location = _map_location

    state = torch.load(str(filename), map_location=map_location)

    model = model_class(**state['model_params'])
    model.load_state_dict(state['model_state'])

    return model

def generate_response(query):
    """
    Generate a response from the model for a given query. The model and the dataset will be taken from the app cache

    :param query: Query to generate the response to

    """

    if not isinstance(query, list):
        query = word_tokenize(query)

    dataset = app_cache['dataset']
    model = app_cache['model']

    query = dataset._process_sent(query)
    query = torch.tensor(query)

    response_logits = model(query.view(1, -1)).squeeze(0)
    response_indices = response_logits.argmax(dim=-1).cpu().numpy()

    response = [dataset.vocab.id2token[int(idx)] for idx in response_indices]
    response = [t for t in response if t not in dataset.vocab.special_tokens]
    response = ' '.join(response)

    return response

### END CODE FROM THE NOTEBOOK ###


app = Flask(__name__)
app.config.from_object(__name__)

app.config.update(dict(
    model_filename='tmp/seq2seq_dialog_att.pt',
    vocab_filename='tmp/seq2seq_dialog.vocab.csv',
    dataset_filename='OpenSubtitles.en.gz',
))
app.config.from_envvar('SEQ2SEQ_DIALOG_SETTINGS', silent=True)


def init_dataset():
    """ Initialize the dataset from the parameters in the app config and return it """
    dataset_filename = app.config['dataset_filename']
    vocab_filename = app.config['vocab_filename']

    vocab = Vocab.load(vocab_filename)
    dataset = SubtitlesDialogDataset(dataset_filename, max_lines=1, vocab=vocab, max_len=50)

    return dataset


def init_model():
    """ Initialize the model from the parameters in the app config and return it """
    model_filename = app.config['model_filename']
    model = load_model(Seq2SeqAttentionModel, model_filename)

    return model


app_cache = dict(
    dataset=init_dataset(),
    model=init_model(),
)


@app.route('/dialog/', methods=['GET'])
def dialog():
    """ Take the query from the GET parameter `query`, generate the reponse, and return a json object """
    query = request.args.get('query')
    response = generate_response(query)

    result = dict(
        query=query,
        response=response,
    )

    return jsonify(**result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
