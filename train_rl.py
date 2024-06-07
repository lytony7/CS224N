import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel
from openai import OpenAI
from torch.optim import Adam

# Function to get OpenAI grade
def get_openai_grade(poem):
    judge_prompt = f"""
Grading Rubric for Poem Generators
Fluency (F)
Definition: The smoothness, readability, and natural flow of the language in the generated poem.

3 (Perfect): The poem reads naturally with no awkward phrases or grammatical errors. It flows smoothly from line to line.
2 (Average): The poem has some noticeable grammatical errors or awkward phrases that affect readability but still maintains overall coherence.
1 (Poor): The poem is difficult to read due to numerous grammatical errors and awkward phrases, making it largely incomprehensible.
Coherence (C)
Definition: The logical and thematic consistency within the poem. How well the lines and stanzas connect to form a unified piece.

3 (Perfect): The poem is logically and thematically consistent throughout. Each line and stanza connect seamlessly, contributing to the overall theme.
2 (Average): The poem has noticeable inconsistencies or weak connections, but the overall theme is still discernible.
1 (Poor): The poem lacks coherence, with lines and stanzas appearing disjointed and unrelated, making the theme unclear.
Meaning (M)
Definition: The depth, clarity, and significance of the message conveyed by the poem.

3 (Perfect): The poem conveys a deep, clear, and meaningful message. It evokes strong emotions or thoughts and has a significant impact.
2 (Average): The poem conveys a message, but it may lack depth or clarity, making it less impactful.
1 (Poor): The poem lacks a clear or meaningful message, making it difficult to understand or connect with.
Aesthetics (A)
Definition: The beauty and artistic quality of the poem, including imagery, metaphor, and other poetic devices.

3 (Perfect): The poem is artistically rich, using vivid imagery, metaphors, and other poetic devices effectively. It is aesthetically pleasing and engaging.
2 (Average): The poem uses some poetic devices and imagery, but they are not particularly vivid or effective.
1 (Poor): The poem lacks any significant use of poetic devices or imagery, making it dull and unengaging.
Overall Evaluation
To evaluate a poem generator, assign scores (1-3) for each of the four categories (Fluency, Coherence, Meaning, Aesthetics). The final score can be an average or a weighted sum based on the importance of each category.

Example:

Fluency: 2
Coherence: 3
Meaning: 2
Aesthetics: 3
Overall Score: (2 + 3 + 2 + 3) / 4 = 2.5

example of perfect poems(total socre=3):
白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
苏武天山上，田横海岛边。万重关塞断，何日是归年。
李白乘舟将欲行，忽闻岸上踏歌声。桃花潭水深千尺，不及汪伦送我情。
朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不尽，轻舟已过万重山。
去年今日此门中，人面桃花相映红。人面不知何处去，桃花依旧笑春风。
十里黄云白日曛，北风吹雁雪纷纷。莫愁前路无知己，天下谁人不识君。
故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，惟见长江天际流。

example of average poems(total score=2):
床前明月光，疑似地上霜。独卧听疏雨，青灯照蛩螀。
凤飞九千仞，五章备彩珍。下有圣人瑞，上合天地珍。
一别家山万里馀，天边消息近何如。梦回忽听檐间鹊，疑是吾儿远寄书。
一别家林二十年，相逢几度各华颠。青山有约常如昔，白首无成独自怜。
江山如此可忘忧，况是经年卧病秋。白日消磨尘满眼，清宵梦寐雨盈舟。


example of poor poems(total score=1):
一雨一月强，雨多泥更狂。马随青草远，人与绿云长。
我爱临封好，宾朋日往还。江喧朝市近，野阔水云閒。
不知谁氏子，亦有一山人。道无南北，相期共苦辛。
山林何敢谓朝簪，野性从来水石心。病起尚能亲药饵，春来还喜入衣襟。

Based on the grading rubic above, grade the following poem
{poem}

Remember just return the overall score. Just return this one number. Do not return anything else.
"""
    client = OpenAI(api_key='sk-proj-d7GugtpJaHRX5djQSlVET3BlbkFJ8BsqFjpzNJ2SmWNQBj4G')

    response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": judge_prompt}
  ]
  )
    final_score_str = response.choices[0].message.content
    final_score = float(final_score_str)
    return final_score

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

# Function to generate poems
def generate_poem(model, tokenizer, n_ctx, length, temperature, top_k, top_p, repetition_penalty, device, is_fast_pattern):
    context = tokenizer.encode("[CLS]", return_tensors='pt').to(device)
    generated = context

    with torch.no_grad():
        while len(generated[0]) < length:
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :]  # Accessing the first element for logits
            next_token_logits = next_token_logits / temperature
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

    # Decode the poem and truncate after the second "。"
    poem_tokens = generated[0].tolist()
    poem = tokenizer.decode(poem_tokens, skip_special_tokens=True).replace(' ', '')

    period_count = 0
    truncated_poem = ""
    for char in poem:
        truncated_poem += char
        if char == '。':
            period_count += 1
        if period_count == 2:
            break

    return truncated_poem

# Reinforcement learning training loop
def train(model, tokenizer, device, args):
    optimizer = Adam(model.parameters(), lr=1e-5)
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for step in range(args.total_steps):
            # Generate poems
            poem = generate_poem(model, tokenizer, model.config.n_ctx, args.length, args.temperature, args.topk, args.topp, args.repetition_penalty, device, args.fast_pattern)
            
            print(f"Generated Poem: {poem}")

            # Get reward
            reward = get_openai_grade(poem)
            print(f"Received Reward: {reward}")

            # Convert poem back to input_ids for loss calculation
            inputs = tokenizer.encode(poem, return_tensors='pt').to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs[0]  # Accessing the first element for loss

            # Policy gradient update
            policy_loss = -loss * reward
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            total_loss += policy_loss.item()

            if (step + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs}, Step {step + 1}/{args.total_steps}, Loss: {total_loss / (step + 1)}")

        # Save model after each epoch
        if not os.path.exists('model/rl_model'):
            os.makedirs('model/rl_model')
        model.save_pretrained('model/rl_model')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--length', default=45, type=int, required=False, help='生成长度')
    parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
    parser.add_argument('--nsamples', default=10, type=int, required=False, help='生成几个样本')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False, help='模型参数')
    parser.add_argument('--tokenizer_path', default='model/final_model/vocab.txt', type=str, required=False, help='词表路径')
    parser.add_argument('--model_path', default='model/rl_model', type=str, required=False, help='模型路径')
    parser.add_argument('--prefix', default='萧炎', type=str, required=False, help='生成文章的开头')
    parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
    parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    parser.add_argument('--fast_pattern', action='store_true', help='采用更加快的方式生成文本')
    parser.add_argument('--save_samples', action='store_true', help='保存产生的样本')
    parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False)
    parser.add_argument('--epochs', default=1, type=int, required=False, help='训练循环')
    parser.add_argument('--total_steps', default=10, type=int, required=False, help='总步数')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)

    train(model, tokenizer, device, args)

if __name__ == '__main__':
    main()
