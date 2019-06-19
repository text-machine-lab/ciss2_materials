from collections import Counter
import nltk
from nltk.tokenize import TweetTokenizer
import numpy as np
import torch.utils.data
import csv
import json
from transformer import Constants

def _read_file(filename):
    history = list()
    response = list()
    ids = list()
    i = 1
    with open(filename, 'r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            if i == 1:
                i = 0
                continue
            ids.append(row[0])
            history.append(row[1].split(" "))
            response.append(row[2].split(" "))

    return history, response, ids



class Vocab(object):
    def __init__(self, special_tokens=None):
        super(Vocab, self).__init__()

        self.nb_tokens = 0

        self.token2id = {}
        self.id2token = {}

        self.token_counts = Counter()

        self.special_tokens = []
        if special_tokens is not None:
            self.special_tokens = special_tokens
            self.add_document(self.special_tokens)

    def add_document(self, document):
        for token in document:
            self.token_counts[token] += 1

            if token not in self.token2id:
                self.token2id[token] = self.nb_tokens
                self.id2token[self.nb_tokens] = token
                self.nb_tokens += 1

    def add_documents(self, documents):
        for doc in documents:
            self.add_document(doc)

    def prune_vocab(self, min_count=2):
        nb_tokens_before = len(self.token2id)

        tokens_to_delete = set([t for t, c in self.token_counts.items() if c < min_count])
        tokens_to_delete -= set(self.special_tokens)

        for token in tokens_to_delete:
            self.token_counts.pop(token)

        self.token2id = {t: i for i, t in enumerate(self.token_counts.keys())}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.nb_tokens = len(self.token2id)

        print('Vocab pruned: {} -> {}'.format(nb_tokens_before, self.nb_tokens))

    def load_from_dict(self, filename):
        with open(filename, 'r') as f:
            self.token2id = json.load(f)
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.nb_tokens = len(self.token2id)

    def save_to_dict(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.token2id, f)

    def __getitem__(self, item):
        return self.token2id[item]

    def __contains__(self, item):
        return item in self.token2id

    def __len__(self):
        return self.nb_tokens

    def __str__(self):
        return 'Vocab: {} tokens'.format(self.nb_tokens)


class DialogueDataset(torch.utils.data.Dataset):
    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    SEP_WORD = '<s>'
    EOS_WORD = '</s>'
    CLS_WORD = '<cls>'

    def __init__(self, filename, history_len = 250, response_len=30, vocab=None, update_vocab=True):
        self.history, self.response, self.ids = _read_file(filename)

        self.history_len = history_len
        self.response_len = response_len

        if vocab is None:
            self.vocab = Vocab(special_tokens=[DialogueDataset.PAD_WORD,
                                               DialogueDataset.UNK_WORD,
                                               DialogueDataset.SEP_WORD,
                                               DialogueDataset.EOS_WORD,
                                               DialogueDataset.CLS_WORD])
        else:
            self.vocab = vocab

        # do not want to update vocab for running old model
        if update_vocab:
            self.vocab.add_documents(self.history)
            self.vocab.add_documents(self.response)

    def _process_history(self, history):
        history = history[-self.history_len+1:]
        history.append(DialogueDataset.EOS_WORD)

        needed_pads = self.history_len - len(history)
        if needed_pads > 0:
            history = history + [DialogueDataset.PAD_WORD] * needed_pads

        history = [
            self.vocab[token] if token in self.vocab else self.vocab[DialogueDataset.UNK_WORD]
            for token in history
        ]

        # create position embeddings, make zero if it is the pad token (0)
        pos = np.array([pos_i+1 if w_i != 0 else 0
            for pos_i, w_i in enumerate(history)])

        #create segment embeddings
        seg = list()
        i = 1
        for j, token in enumerate(history):
            if token == self.vocab[DialogueDataset.PAD_WORD]:
                break
            seg.append(i)
            if token == self.vocab[DialogueDataset.SEP_WORD]:
                i+=1
        seg += [0] * needed_pads
        seg = np.array(seg, dtype=np.long)

        history = np.array(history, dtype=np.long)

        return (history, pos, seg)

    def _process_response(self, response):
        response = response[:self.response_len - 1]
        response.append(DialogueDataset.EOS_WORD)
        #response.insert(0, DialogueDataset.CLS_WORD)

        needed_pads = self.response_len - len(response)
        if needed_pads > 0:
            response = response + [DialogueDataset.PAD_WORD] * needed_pads

        response = [
            self.vocab[token] if token in self.vocab else self.vocab[DialogueDataset.UNK_WORD]
            for token in response
        ]
        # create position embeddings
        pos = np.array([pos_i + 1 if w_i != 0 else 0
                        for pos_i, w_i in enumerate(response)])
        response = np.array(response, dtype=np.long)
        return (response, pos)
    
    def get_input_features(self, history):
        tokenizer = TweetTokenizer()
        all_history = list()
        all_history.append(DialogueDataset.CLS_WORD)
        for line in history:
            all_history+=list(tokenizer.tokenize(line))
            all_history.append(DialogueDataset.SEP_WORD)
        examples = self._process_history(all_history[:-1])
        return torch.from_numpy(examples[0]).unsqueeze(0), torch.from_numpy(examples[1]).unsqueeze(0), torch.from_numpy(examples[2]).unsqueeze(0)

    def __getitem__(self, index):
        history = self._process_history(self.history[index])
        response = self._process_response(self.response[index])
        id = self.ids[index]
        return history[0], history[1], history[2], response[0], response[1]

    def __len__(self):
        return len(self.history)
