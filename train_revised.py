import transformers
import torch
import os
import json
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

def build_files(data_path, tokenized_data_path, num_pieces, tokenizer, min_length):
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = [line.replace('\n', ' [SEP] ') for line in lines if len(line.strip()) > min_length]
    all_len = len(lines)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (i + 1):])
        sublines = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line)) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.append(tokenizer.convert_tokens_to_ids('[MASK]'))
            full_line.extend(subline)
            full_line.append(tokenizer.convert_tokens_to_ids('[CLS]'))
        with open(f'{tokenized_data_path}tokenized_train_{i}.txt', 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
    print('finish')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='Set CUDA devices')
    parser.add_argument('--model_config', default='gpt2', type=str, help='Model config')
    parser.add_argument('--tokenizer_path', default='gpt2', type=str, help='Tokenizer path')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, help='Raw data path')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, help='Tokenized data path')
    parser.add_argument('--raw', action='store_true', help='Whether to tokenize raw data')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, help='Learning rate')
    parser.add_argument('--warmup_steps', default=2000, type=int, help='Warmup steps')
    parser.add_argument('--log_step', default=100, type=int, help='Logging step interval')
    parser.add_argument('--stride', default=768, type=int, help='Stride length for training data')
    parser.add_argument('--gradient_accumulation', default=1, type=int, help='Gradient accumulation steps')
    parser.add_argument('--fp16', action='store_true', help='Use FP16')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, help='FP16 optimization level')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Max gradient norm')
    parser.add_argument('--num_pieces', default=100, type=int, help='Number of pieces for training data')
    parser.add_argument('--min_length', default=128, type=int, help='Minimum length of input sequences')
    parser.add_argument('--output_dir', default='model/', type=str, help='Output directory')
    parser.add_argument('--pretrained_model', default='', type=str, help='Path to pretrained model')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, help='Tensorboard directory')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.pretrained_model or args.model_config)

    if args.raw:
        build_files(args.raw_data_path, args.tokenized_data_path, args.num_pieces, tokenizer, args.min_length)

    model.train()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * (sum([len(open(f'{args.tokenized_data_path}tokenized_train_{i}.txt').read().strip().split()) for i in range(args.num_pieces)]) // args.stride) // args.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    tb_writer = SummaryWriter(log_dir=args.writer_dir)

    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        now = datetime.now()
        x = np.linspace(0, args.num_pieces - 1, args.num_pieces, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0
        for i in x:
            with open(f'{args.tokenized_data_path}tokenized_train_{i}.txt', 'r') as f:
                tokens = [int(token) for token in f.read().strip().split()]
            start_point = 0
            samples = []
            while start_point < len(tokens) - model.config.n_ctx:
                samples.append(tokens[start_point: start_point + model.config.n_ctx])
                start_point += args.stride
            if start_point < len(tokens):
                samples.append(tokens[len(tokens)-model.config.n_ctx:])
            random.shuffle(samples)
            for step in range(len(samples) // args.batch_size):
                batch = samples[step * args.batch_size: (step + 1) * args.batch_size]
                batch_inputs = torch.tensor(batch).long().to(device)

                outputs = model(input_ids=batch_inputs, labels=batch_inputs)
                loss = outputs.loss

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if (step + 1) % args.gradient_accumulation == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if (step + 1) % args.log_step == 0:
                    tb_writer.add_scalar('loss', loss.item(), epoch * len(samples) // args.batch_size + step)
                    print(f'Step {step + 1}, Loss: {loss.item()}')

        output_dir = os.path.join(args.output_dir, f'epoch_{epoch + 1}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    final_output_dir = os.path.join(args.output_dir, 'final_model')
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print('Training completed')

if __name__ == '__main__':
    main()
