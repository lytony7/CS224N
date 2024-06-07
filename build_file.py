import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from tokenizations.bpe_tokenizer import get_encoder
from tokenizations import tokenization_bert

def build_files(data_path, tokenized_data_path, num_pieces, full_tokenizer):
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)

    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)

    for i in tqdm(range(num_pieces)):
        sublines = lines[len(lines) // num_pieces * i: len(lines) // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines[len(lines) // num_pieces * (i + 1):])  # Add remaining lines to the last piece

        tokenized_inputs = []
        tokenized_targets = []
        for line in sublines:
            input_line = line['Input']
            target_line = line['Target']
            if '□' in input_line or '□' in target_line:
                continue  # Skip lines containing "□"
            input_ids = full_tokenizer.encode(input_line)
            target_ids = full_tokenizer.encode(target_line)
            tokenized_inputs.append(input_ids)
            tokenized_targets.append(target_ids)

        if not tokenized_inputs or not tokenized_targets:
            print(f"No valid tokenized inputs/targets for piece {i}")
            continue

        full_line = []
        for input_ids, target_ids in zip(tokenized_inputs, tokenized_targets):
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # Add MASK at the beginning
            full_line.extend(input_ids)
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # Add CLS at the end
            full_line.extend(target_ids)
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # Add CLS between input and target

        with open(os.path.join(tokenized_data_path, f'tokenized_train_{i}.txt'), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
        print(f"Finished processing piece {i}")
    print('finish')

full_tokenizer = tokenization_bert.BertTokenizer(vocab_file='model/self_train_model_2/vocab.txt')

build_files(data_path='data/sw_traindata.json', tokenized_data_path='data/tokenized/', num_pieces=100,
                    full_tokenizer=full_tokenizer)
