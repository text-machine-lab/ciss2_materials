import numpy as np
import torch
from torch.autograd import Variable
import os
import json
from torch.nn import Parameter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate import bleu_score
import torch.nn.functional as F
import transformer

def get_sequences_lengths(sequences, masking=0, dim=1):
    if len(sequences.size()) > 2:
        sequences = sequences.sum(dim=2)

    masks = torch.ne(sequences, masking)

    lengths = masks.sum(dim=dim)

    return lengths

def get_sentence_from_indices(indices, vocab, eos_token, join=True):
    tokens = []
    for idx in indices:
        token = vocab.id2token[idx]

        if token == eos_token:
            break

        tokens.append(token)

    if join:
        tokens = ' '.join(tokens)

    tokens = tokens

    return tokens


def get_pretrained_embeddings(embeddings_dir):
    embeddings = np.load(embeddings_dir)
    emb_tensor = torch.FloatTensor(embeddings)
    return emb_tensor

def save_args(filename, args):
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)


def save_checkpoint(filename, model, optimizer):
    '''
    saves model into a state dict, along with its training statistics,
    and parameters
    :param model:
    :param optimizer:
    :return:
    '''
    state = {
        'model': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        }
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer, device):
    '''
    loads previous model
    :param filename: file name of model
    :param model: model that contains same parameters of the one you are loading
    :param optimizer:
    :return: loaded model, checkpoint
    '''
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def freeze_layer(layer, bool):
    '''
    freezes or unfreezes layer of pytorch model
    :param layer:
    :param bool: True to freeze, False to unfreeze
    :return:
    '''
    for param in layer.parameters():

        param.requires_grad = not bool
    layer.training = not bool
    return layer


def bleu(targets, predicted, n_grams=4):
    '''
    calculates bleu score
    :param targets: tensor of actual sentences
    :param predicted: tensor of predicted sentences
    :param n_grams: number of n-grams for bleu score (int)
    :return: bleu score (float)
    '''
    reference = [[[str(x.item())for x in row if x.item() != 0]]
                 for row in targets]
    hypothosis = [[str(x.item()) for x in row if x.item() != 0]
                 for row in predicted]
    weights = [1/n_grams] * n_grams

    chencherry = bleu_score.SmoothingFunction()
    bleu_1 = bleu_score.corpus_bleu(
        reference, hypothosis, weights=weights,
        smoothing_function=chencherry.method1)
    return bleu_1

