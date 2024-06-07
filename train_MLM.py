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


def build_files_mlm(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length, mask_prob=0.15):
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line.replace('\n', ' [SEP] ') for line in lines]

    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)

    for i in tqdm(range(num_pieces)):
        sublines = lines[len(lines) // num_pieces * i: len(lines) // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines[len(lines) // num_pieces * (i + 1):])

        tokenized_lines = [full_tokenizer.tokenize(line) for line in sublines if len(line) > min_length]
        tokenized_lines = [full_tokenizer.convert_tokens_to_ids(line) for line in tokenized_lines]

        full_lines = []
        for line in tokenized_lines:
            input_ids = []
            labels = []
            for token in line:
                if random.random() < mask_prob:
                    input_ids.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))
                    labels.append(token)
                else:
                    input_ids.append(token)
                    labels.append(-100)  # -100 for tokens we don't want to predict
            full_lines.append((input_ids, labels))

        with open(f'{tokenized_data_path}/tokenized_train_{i}.txt', 'w') as f:
            for input_ids, labels in full_lines:
                f.write(' '.join(map(str, input_ids)) + '\n')
                f.write(' '.join(map(str, labels)) + '\n')
        print(f"Finished processing piece {i}")
    print('finish')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='Set which GPUs to use')
    parser.add_argument('--model_config', default='model/self_train_model_2/config.json', type=str, required=False, help='Model config path')
    parser.add_argument('--tokenizer_path', default='model/self_train_model_2/vocab.txt', type=str, required=False, help='Tokenizer path')
    parser.add_argument('--raw_data_path', default='data/sw_traindata.json', type=str, required=False, help='Raw training data path')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False, help='Tokenized data path')
    parser.add_argument('--raw', action='store_true', help='Whether to tokenize raw data first')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='Number of epochs')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='Batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='Learning rate')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='Warm-up steps')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='Log step interval')
    parser.add_argument('--stride', default=768, type=int, required=False, help='Training stride')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='Gradient accumulation')
    parser.add_argument('--fp16', action='store_true', help='Mixed precision training')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='Number of data pieces')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='Minimum length of sequences')
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='Model output path')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='Pretrained model path')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard path')
    parser.add_argument('--segment', action='store_true', help='Tokenize Chinese by words')
    parser.add_argument('--bpe_token', action='store_true', help='Use BPE tokenization')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="Encoder json path")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="Vocab bpe path")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_config = transformers.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    if args.bpe_token:
        full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
    else:
        full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if raw:
        print('building files')
        build_files_mlm(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, num_pieces=num_pieces,
                        full_tokenizer=full_tokenizer, min_length=min_length)
        print('files built')

    if not args.pretrained_model:
        model = transformers.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()
    model.to(device)

    num_parameters = sum(p.numel() for p in model.parameters())
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    full_len = 0
    print('calculating total steps')
    for i in tqdm(range(num_pieces)):
        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
            full_len += len([int(item) for item in f.read().strip().split()])
    total_steps = int(full_len / stride * epochs / batch_size / gradient_accumulation)
    print('total steps = {}'.format(total_steps))

    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    print('starting training')
    overall_step = 0
    running_loss = 0
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0
        for i in x:
            with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
                lines = f.read().strip().split('\n')
                inputs = [list(map(int, line.split())) for line in lines[0::2]]
                labels = [list(map(int, line.split())) for line in lines[1::2]]
            total_length = len(inputs)
            assert total_length == len(labels)
            steps = (total_length - 1) // stride + 1
            for step in range(steps):
                overall_step += 1
                batch_inputs = []
                batch_labels = []
                for j in range(batch_size):
                    if step * stride + j < total_length:
                        batch_inputs.append(inputs[step * stride + j])
                        batch_labels.append(labels[step * stride + j])
                batch_inputs = torch.tensor(batch_inputs).to(device)
                batch_labels = torch.tensor(batch_labels).to(device)
                outputs = model(input_ids=batch_inputs, labels=batch_labels)
                loss = outputs.loss
                if multi_gpu:
                    loss = loss.mean()
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                running_loss += loss.item()
                if overall_step % log_step == 0:
                    tb_writer.add_scalar('loss', running_loss / log_step, overall_step)
                    print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        step + 1,
                        piece_num,
                        epoch + 1,
                        running_loss / log_step))
                    running_loss = 0
                if (step + 1) % gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            piece_num += 1
            if piece_num % 10 == 0:
                output_model_path = output_dir + 'model_epoch{}_piece{}.bin'.format(epoch + 1, piece_num)
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), output_model_path)
                print('Model saved to {}'.format(output_model_path))
    tb_writer.close()
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_path = output_dir + 'model_epoch{}.bin'.format(epoch + 1)
    torch.save(model_to_save.state_dict(), output_model_path)
    print('Model saved to {}'.format(output_model_path))


if __name__ == '__main__':
    main()
