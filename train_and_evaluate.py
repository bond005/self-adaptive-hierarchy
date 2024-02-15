from argparse import ArgumentParser
import logging
import os
import random

from datasets import load_dataset
import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding

from neural_network.hierarchical_bert import BertForHierarchicalEmbedding
from neural_network.hierarchical_bert import BertForHierarchicalSequenceClassification
from neural_network.hierarchical_bert import HierarchicalBertConfig


RANDOM_SEED: int = 42
BERT_NAME = 'google-bert/bert-base-cased'
tokenizer: BertTokenizer


def mnli_to_array_fn(batch):
    premise = batch['premise']
    hypothesis = batch['hypothesis']
    input_ids = tokenizer.encode(
        tokenizer.cls_token + premise + tokenizer.sep_token + hypothesis + tokenizer.sep_token,
        add_special_tokens=False
    )
    batch['input_ids'] = input_ids


def mrpc_to_array_fn(batch):
    sentence1 = batch['sentence1']
    sentence2 = batch['sentence2']
    input_ids = tokenizer.encode(
        tokenizer.cls_token + sentence1 + tokenizer.sep_token + sentence2 + tokenizer.sep_token,
        add_special_tokens=False
    )
    batch['input_ids'] = input_ids


def sst2_to_array_fn(batch):
    sentence = batch['sentence']
    input_ids = tokenizer.encode(
        sentence,
        add_special_tokens=True
    )
    batch['input_ids'] = input_ids


def main():
    parser = ArgumentParser()
    parser.add_argument('-b', '--base', dest='base_model_name', type=str, required=False,
                        default=BERT_NAME, help='The base BERT name.')
    parser.add_argument('-t', '--task', dest='task_name', type=str, required=True,
                        choices=['sst2', 'mnli', 'mrpc'], help='The solved task in the GLUE benchmark.')
    parser.add_argument('-r', '--restarts', dest='restarts_number', type=int, required=True,
                        help='The number of experiment restarts with different random seeds.')
    parser.add_argument('--siamese', dest='use_siamese_tuning', action='store_true',
                        help='Is the "siamese" training stage in the fine-tuning process used?')
    parser.add_argument('--hierarchy', dest='use_hierarchy', action='store_true',
                        help='Is the self-adaptive hierarchical BERT used instead of the usual BERT?')
    args = parser.parse_args()

    training_set = load_dataset('glue', args.task_name, split='train')
    validation_set = load_dataset('glue', args.task_name, split='validation')
    test_set = load_dataset('glue', args.task_name, split='test')
    print(f'The dataset "{args.task_name}" is loaded.')

    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.base_model_name)
    print(f'The BERT tokenizer is loaded from "{args.base_model_name}".')
    pass


if __name__ == '__main__':
    main()
