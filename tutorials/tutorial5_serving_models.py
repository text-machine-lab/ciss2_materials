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
    END_TOKEN = '<end>'
    START_TOKEN = '<start>'
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    def __init__(self, special_tokens=None):
        super().__init__()

        self.special_tokens = special_tokens

        self.token2id = {}
        self.id2token = {}

        self.token_counts = Counter()

        if self.special_tokens is not None:
            self.add_document(self.special_tokens)

    def add_document(self, document, rebuild=True):
        for token in document:
            self.token_counts[token] += 1

            if token not in self.token2id:
                self.token2id[token] = len(self.token2id)

        if rebuild:
            self._rebuild_id2token()

    def add_documents(self, documents):
        for doc in documents:
            self.add_document(doc, rebuild=False)

        self._rebuild_id2token()

    def prune_vocab(self, max_size):
        nb_tokens_before = len(self.token2id)

        tokens_all = set(self.token2id.keys())
        tokens_special = set(self.special_tokens)
        tokens_most_common = set(t for t, c in self.token_counts.most_common(max_size)) - tokens_special
        tokens_to_delete = tokens_all - tokens_most_common - tokens_special

        for token in tokens_to_delete:
            self.token_counts.pop(token)

        self.token2id = {}
        for i, token in enumerate(self.special_tokens):
            self.token2id[token] = i
        for i, token in enumerate(tokens_most_common):
            self.token2id[token] = i + len(self.special_tokens)

        self._rebuild_id2token()

        nb_tokens_after = len(self.token2id)

        print(f'Vocab pruned: {nb_tokens_before} -> {nb_tokens_after}')

    def _rebuild_id2token(self):
        self.id2token = {i: t for t, i in self.token2id.items()}

    def get(self, item, default=None):
        return self.token2id.get(item, default)

    def __getitem__(self, item):
        return self.token2id[item]

    def __contains__(self, item):
        return item in self.token2id

    def __len__(self):
        return len(self.token2id)

    def __str__(self):
        return f'{len(self)} tokens'

    def save(self, filename):
        with open(filename, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['token', 'counts', 'is_special'])
            writer.writeheader()
            for idx in range(len(self.token2id)):
                token = self.id2token[idx]
                is_special = 1 if token in self.special_tokens else 0
                writer.writerow({'token': token, 'counts': self.token_counts[token], 'is_special': is_special})

    @staticmethod
    def load(filename):
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
    def __init__(self, filename, vocab=None, max_lines=1000, max_len=50, max_vocab_size=50000):

        self.lines = []
        with gzip.open(filename, 'rb') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break

                self.lines.append(word_tokenize(line.decode('utf-8')))

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
        sent = sent[:self.max_len - 1] + [Vocab.END_TOKEN, ]

        nb_pad = self.max_len - len(sent)
        sent = sent + [Vocab.PAD_TOKEN, ] * nb_pad

        return sent

    def _process_sent(self, sent):
        sent = self._pad_sentnece(sent)
        sent = [self.vocab[t] if t in self.vocab else self.vocab[Vocab.UNK_TOKEN] for t in sent]

        sent = np.array(sent, dtype=np.long)
        return sent

    def __getitem__(self, index):
        query = self.lines[index]
        response = self.lines[index+1]

        query = self._process_sent(query)
        response = self._process_sent(response)

        return query, response

    def __len__(self):
        return self.max_lines - 1

def softmax_masked(inputs, mask, dim=1, epsilon=0.000001):
    inputs_exp = torch.exp(inputs)
    inputs_exp = inputs_exp * mask.float()
    inputs_exp_sum = inputs_exp.sum(dim=dim)
    inputs_attention = inputs_exp / (inputs_exp_sum.unsqueeze(dim) + epsilon)

    return inputs_attention

class Seq2SeqAttentionModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, teacher_forcing,
                 max_len,trainable_embeddings, start_index, end_index, pad_index, W_emb=None):

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
        batch_size = inputs.size(0)
        inputs_mask = (inputs != self.pad_index).long()
        inputs_lengths = torch.sum(inputs_mask, dim=1)
        
        inputs_emb = self.embedding(inputs)
        outputs, h = self.encoder(inputs_emb)
        
#         h_last_hidden = outputs[np.arange(batch_size), inputs_lengths - 1]
        
        return outputs, inputs_mask
    
    def decode(self, encoder_hiddens, inputs_mask, targets=None):
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
        encoder_hiddens, inputs_mask = self.encode(inputs)
        outputs_logits = self.decode(encoder_hiddens, inputs_mask, targets)

        return outputs_logits

def load_model(model_class, filename):
    def _map_location(storage, loc):
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
    dataset_filename = app.config['dataset_filename']
    vocab_filename = app.config['vocab_filename']

    vocab = Vocab.load(vocab_filename)
    dataset = SubtitlesDialogDataset(dataset_filename, max_lines=1, vocab=vocab, max_len=50)

    return dataset


def init_model():
    model_filename = app.config['model_filename']
    model = load_model(Seq2SeqAttentionModel, model_filename)

    return model


app_cache = dict(
    dataset=init_dataset(),
    model=init_model(),
)


@app.route('/dialog/', methods=['GET'])
def dialog():
    query = request.args.get('query')
    response = generate_response(query)

    result = dict(
        query=query,
        response=response,
    )

    return jsonify(**result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
