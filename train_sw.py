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

# The rest of your main function
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
    parser.add_argument('--pretrained_model', default='model/self_train_model_2', type=str, required=False, help='Pretrained model path')
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

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # Set GPUs to use

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
    raw = args.raw  # Whether to tokenize raw data first
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # Do not enable if your GPU does not support half precision
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
        build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, num_pieces=num_pieces,
                    full_tokenizer=full_tokenizer)
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
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
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
                line = f.read().strip()
            tokens = line.split()
            tokens = [int(token) for token in tokens]
            start_point = 0
            samples = []
            while start_point < len(tokens) - n_ctx:
                samples.append(tokens[start_point: start_point + n_ctx])
                start_point += stride
            if start_point < len(tokens):
                samples.append(tokens[len(tokens)-n_ctx:])
            random.shuffle(samples)
            for step in range(len(samples) // batch_size):  # drop last

                #  prepare data
                batch = samples[step * batch_size: (step + 1) * batch_size]
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                batch_inputs = torch.tensor(batch_inputs).long().to(device)

                #  forward pass
                outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                loss, logits = outputs[:2]

                #  get loss
                if multi_gpu:
                    loss = loss.mean()
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation

                #  loss backward
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                #  optimizer step
                if (overall_step + 1) % gradient_accumulation == 0:
                    running_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                if (overall_step + 1) % log_step == 0:
                    tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                    print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        step + 1,
                        piece_num,
                        epoch + 1,
                        running_loss * gradient_accumulation / (log_step / gradient_accumulation)))
                    running_loss = 0
                overall_step += 1
            piece_num += 1

        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    main()
