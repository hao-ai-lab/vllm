import gradio as gr
import requests
import time
import json
from transformers import AutoTokenizer
from openai import OpenAI
from evaluator import extract_answer, strip_string, math_equal, extract_first_boxed_answer


##################################Aux functions
import math
from collections import Counter, defaultdict
from typing import List
import numpy as np

def entropy(Plist):
    if len(Plist):
        result = 0
        for x in Plist:
            result += (-x) * math.log(x, 2)
        return result
    else:
        return 0

def norm(Olist):
    s = sum(Olist)
    return [o / s for o in Olist]

def count(Olist):
    x_dict = defaultdict(lambda: 0.0)
    for x in Olist:
        x_dict[x] += 1
    cc = [c for _,c in x_dict.items()]
    #print(cc)
    return cc

def item_entropy(answers: List) -> float:
    return entropy(norm(count(answers)))

def count_not_empty(answers):
    return sum(1 for answer in answers if answer != '')

def eqaul_group(answers):
    equiv_classes = []
    
    
    for answer in answers:
        weight = 1
        flag = 0
        for i, rep in enumerate(equiv_classes):
            if math_equal(answer,rep):
                flag = 1
                break
        if flag:
            continue
        equiv_classes.append(answer)
        
    return len(equiv_classes) == 1


def majority_voting(answers):
    equiv_classes = []
    equiv_weights = []
    max_vote = 0
    for answer in answers:
        weight = 1
        flag = 0
        for i, rep in enumerate(equiv_classes):
            if math_equal(answer,rep):
                flag = 1
                equiv_weights[i] = equiv_weights[i]+weight
                if equiv_weights[i] > max_vote:
                    max_vote = equiv_weights[i]
                    max_rep = answer
                break
        if flag:
            continue
        equiv_classes.append(answer)
        equiv_weights.append(weight)
        if max_vote == 0:
            max_vote = weight
            max_rep = answer
    return max_rep

def obtaint_answer(s):
    # Find first unpaired } by counting { and }
    stack = []
    for i, c in enumerate(s):
        if c == '{':
            stack.append(c)
        elif c == '}':
            if not stack:  # No matching { found
                return s[:i]
            stack.pop()
    return ""
##################################Aux functions


openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8000/v1"
model = '/workspace/model/DeepSeek-R1-Distill-Qwen-7B'
# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def predict(prompt_dropdown, message, history_baseline, history_adaptive):

    print(f"{prompt_dropdown = }")
    print(f"{message = }")

    if isinstance(prompt_dropdown, dict):
        prompt_dropdown = prompt_dropdown['prompt']
    if isinstance(message, dict):
        message = message['prompt']
    prompt = message or prompt_dropdown
    print(f"{prompt = }")
    message = prompt

    #print('History: ', history_baseline)
    print('==========================')
    print('Only Support Single Round')
    print('==========================')
    
    history_baseline, history_adaptive = [], []
    
    # Convert chat history to OpenAI format
    history_openai_format_baseline = []
    message_sending = ""
    for human, assistant in history_baseline:
        print('Should be empty')
        message_sending += '<｜User｜>' + human
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({
            "role": "assistant",
            "content": assistant
        })
        message_sending += '<｜Assistant｜>' + assistant
    history_openai_format_baseline.append({"role": "user", "content": prompt})
    message_sending += '<｜User｜>' + message + '<｜Assistant｜>'
    message_display = ""
        
    # Create a chat completion request and send it to the API server
    # stream = client.chat.completions.create(
    #     model=model,  # Model name to use
    #     messages=history_openai_format,  # Chat history
    #     temperature=0.7,  # Temperature for text generation
    #     stream=True, max_tokens=16384) # Stream response

    
    step = 0
    warmup_steps = 0
    continue_certain_bar = 3
    max_len = 12000
    adaptive_end = False
    append_answer = False
    uncertain_words = ['wait', 'hold', 'but', 'okay', 'no', 'hmm']
    answers, responses = [], []
    print(history_baseline)
    history_baseline.append((prompt, ""))
    history_adaptive.append((prompt, ""))

    start_ts = time.time()
    latency_baseline_value = ""
    latency_adaptive_value = ""

    print('Message Sending: ', message_sending)
    while True:
        completion = client.completions.create(
            model=model,
            temperature=0.6,
            prompt=message_sending,
            stream=True, max_tokens=64, top_p=0.95,
        )

        probe = client.completions.create(
            model=model,
            temperature=0.6,
            prompt=message_sending + '... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{',
            stream=True, max_tokens=20, top_p=0.95,
        )

        partial_message = ""
        for chunk in completion:
            #print(chunk.choices[0].finish_reason)
            new_text = (chunk.choices[0].text or "")
            partial_message += new_text
            message_display += new_text
            display_message = message_display.replace('<think>', '[think]').replace('</think>', '[/think]')
            history_baseline[-1] = (message, display_message)
            if not adaptive_end:
                history_adaptive[-1] = (message, display_message)

            #print('Finish: ', chunk.choices[0].finish_reason)
            cur_ts = time.time()
            latency_baseline_value = cur_ts - start_ts
            if not adaptive_end:
                latency_adaptive_value = cur_ts - start_ts
            speedup_value = latency_baseline_value / latency_adaptive_value
            
            
            num_token_baseline = len(tokenizer.encode(history_baseline[-1][1]))
            num_token_adaptive = len(tokenizer.encode(history_adaptive[-1][1]))
            # print(f"{num_token_baseline = }, {num_token_adaptive = }")

            token_saved_value = abs(num_token_baseline - num_token_adaptive)
            yield (
                "",
                history_baseline,
                history_adaptive,
                f"{latency_baseline_value:.2f}",
                f"{latency_adaptive_value:.2f}",
                f"{speedup_value:.2f} x",
                f"{token_saved_value}"
            )

        message_sending += partial_message
        if chunk.choices[0].finish_reason is not None and chunk.choices[0].finish_reason != 'length':
            break

        probe_text = ''
        for probe_chunk in probe:
            probe_text += probe_chunk.choices[0].text
        step += 1
        print('Step: ', step, max_len, max_len // 64)
        if step >= max_len // 64:
            break
        
        answer = obtaint_answer(probe_text)    
        answers.append(answer)
        responses.append(probe_text)

        certain_count = [not any(word in res.lower() for word in uncertain_words) for res in responses[-continue_certain_bar:]]

        print("=" * 100)
        print(probe_text, answer, certain_count)
        print("=" * 100)
        
        if step >= warmup_steps and eqaul_group(answers[-continue_certain_bar:]) and count_not_empty(answers[-continue_certain_bar:]) == continue_certain_bar and sum(certain_count) == continue_certain_bar :
            adaptive_end = True

        if adaptive_end and not append_answer:
            print('Adaptive Ending')
            append_answer = True
            if '[/think]' in history_adaptive[-1][-1]:
                history_adaptive[-1] = (message, history_adaptive[-1][-1] + '\n\n... Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{' + answers[-1] + '}\n\\]')
            else:
                history_adaptive[-1] = (message, history_adaptive[-1][-1] + '\n\n...[/think]\n Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{' + answers[-1] + '}\n\\]')
# extra_body={
        #     'repetition_penalty':
        #     1,
        #     'stop_token_ids': [
        #         int(id.strip()) for id in args.stop_token_ids.split(',')
        #         if id.strip()
        #     ] if args.stop_token_ids else []
        # })

    # Read and return generated text from response stream
    # partial_message = ""
    # print(history_baseline)
    # history_baseline.append((message, partial_message))
    # history_adaptive.append((message, partial_message))

    # #cnt = 0
    # for chunk in stream:
    #     partial_message += (chunk.choices[0].delta.content or "")
    #     display_message = partial_message.replace('<think>', '[think]').replace('</think>', '[/think]')
    #     history_baseline[-1] = (message, display_message)
    #     history_adaptive[-1] = (message, display_message)
    #     #cnt += 1
    #     #print(history_baseline, history_adaptive, cnt)
    #     yield "", history_baseline, history_adaptive
    #     #yield "", [(message, "hello world " + str(cnt))], [(message, "hello world"  + str(cnt))] # history_baseline, history_adaptive


# Define categorized prompts
PROMPT_CATEGORIES = {
    "Quick Test" : [
        {"prompt": "Solve 1+1", "answer": "2"},
        {"prompt": "Solve this calculus problem: find the derivative of x^3 + 2x^2 + 5x + 3",
         "answer": "3x^2 + 4x + 5"}
    ]
}

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def process_AMC():
    top10 = [35, 39,  5,  9, 19, 13, 37, 17,  2,  3] 
    top30 = [34,  6,  8, 27, 11, 28, 15, 18, 20, 31] 
    top10_30 = [14, 12, 32,  4, 25, 21, 30,  0,  1,  7, 22, 24, 23, 33, 29, 10, 36, 26, 16, 38] 
    PROMPT_CATEGORIES['AMC23 - Easy'] = []
    PROMPT_CATEGORIES['AMC23 - Median'] = []
    PROMPT_CATEGORIES['AMC23 - Hard'] = []
    amc23 = load_jsonl('test-amc23.jsonl')
    for i in range(40):
        if i in top10:
            PROMPT_CATEGORIES['AMC23 - Easy'].append({"prompt": amc23[i]['problem'], "answer": str(amc23[i]['answer'])})
        elif i in top30:
            PROMPT_CATEGORIES['AMC23 - Hard'].append({"prompt": amc23[i]['problem'], "answer": str(amc23[i]['answer'])})
        elif i in top10_30:
            PROMPT_CATEGORIES['AMC23 - Median'].append({"prompt": amc23[i]['problem'], "answer": str(amc23[i]['answer'])})
        else:
            assert False, f" found {i}"
def process_MATH500():
    top10 = [184,128,83,21,302,1,349,210,152,136] 
    top30 = [444,383,36,240,264,96,400,257,422,456] 
    top10_30 = [203,211,91,399,243,127,3,27,134,40,267,98,411,363,488,42,404,473,270,132] 

    short = [455,207,77,54,161,122,31,262,255,8]
    short_0_7 = [77,255,229,185,135,265,148,215,196,350]

    short_0_7_3_shortest = [399,406,457,13,430,191,207,105,455,374]
    short_0_8_3_500 = [397,77,86,161,413,56,117,262,5,31]
    PROMPT_CATEGORIES['MATH500 - Short 0.7'] = []
    PROMPT_CATEGORIES['MATH500 - Short >500'] = []

    # PROMPT_CATEGORIES['MATH500 - Easy'] = []
    # PROMPT_CATEGORIES['MATH500 - Median'] = []
    # PROMPT_CATEGORIES['MATH500 - Hard'] = []
    amc23 = load_jsonl('test-math500.jsonl')
    for i in range(500):
        if i in short_0_7_3_shortest:
            PROMPT_CATEGORIES['MATH500 - Short 0.7'].append({"prompt": amc23[i]['problem'], "answer": str(amc23[i]['answer'])})
        if i in short_0_8_3_500:
            PROMPT_CATEGORIES['MATH500 - Short >500'].append({"prompt": amc23[i]['problem'], "answer": str(amc23[i]['answer'])})

# if i in top10:
        #     PROMPT_CATEGORIES['MATH500 - Easy'].append({"prompt": amc23[i]['problem'], "answer": str(amc23[i]['answer'])})
        # elif i in top30:
        #     PROMPT_CATEGORIES['MATH500 - Hard'].append({"prompt": amc23[i]['problem'], "answer": str(amc23[i]['answer'])})
        # elif i in top10_30:
        #     PROMPT_CATEGORIES['MATH500 - Median'].append({"prompt": amc23[i]['problem'], "answer": str(amc23[i]['answer'])})
        # else:
        #     pass





# process_AMC()

process_MATH500()
def update_prompt_options(category):
    """Update prompt options based on selected category"""
    return gr.update(choices=[p['prompt'] for p in PROMPT_CATEGORIES.get(category, [])], visible=True)
    
def update_reference_answer(prompt):
    if isinstance(prompt, dict):
        return prompt['answer']

    # 在所有类别中查找该提示词对应的答案
    for category in PROMPT_CATEGORIES.values():
        for cat in category:
            if prompt in cat["prompt"]:
                return cat["answer"]
    return ""  
    
# Create Gradio interface
with gr.Blocks(css="footer {visibility: hidden}", theme=gr.themes.Citrus()) as demo:
    gr.Markdown("# LLM Model Comparison: Baseline vs. Adaptive Compute")
    
    with gr.Row():
        with gr.Column():
            chatbot_baseline = gr.Chatbot(label="Model A (Baseline)")

        with gr.Column():
            chatbot_adaptive = gr.Chatbot(label="Model B (Adaptive Compute)")

    with gr.Row():
        with gr.Column():
            latency_baseline = gr.Textbox(
                label="Latency (Baseline)", interactive=False
            )
            
        with gr.Column():
            latency_adaptive = gr.Textbox(
                label="Latency (Adaptive Compute)", interactive=False
            )
    with gr.Row():
        # speedup
        with gr.Column():
            speedup = gr.Textbox(
                label="Speedup", interactive=False
            )
        with gr.Column():
            token_saved = gr.Textbox(
                label="Token Saved", interactive=False
            )

    
    gr.Markdown("Select a (category, prompt) pair:")
    with gr.Row():
        category_dropdown = gr.Dropdown(
            choices=list(PROMPT_CATEGORIES.keys()),
            label="Select Category",
            value="Quick Test",
            allow_custom_value=True
        )
        prompt_dropdown = gr.Dropdown(
            choices=[d['prompt'] for d in PROMPT_CATEGORIES["Quick Test"]],
            label="Select Prompt",
            value=PROMPT_CATEGORIES["Quick Test"][1]['prompt'],
            interactive=True,
            allow_custom_value=True
        )
        reference_answer = gr.Textbox(
            label="Reference Answer",
            interactive=False,
            value="3x^2 + 4x + 5",
            scale=1,
            lines=1,
            # allow_custom_value=True
        )

    gr.Markdown("or enter your prompt directly:")
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Enter your prompt or select from categories below",
            container=False,
            scale=3  # Takes up 3 parts of the 4 total parts
        )
    
    submit_btn = gr.Button("Submit", scale=1, elem_classes="orange-button")
    
    with gr.Row():
        with gr.Column():
            random_btn = gr.Button("🎲 Random Example")
        with gr.Column():
            clear_btn = gr.Button("🗑️ Clear history")
        # with gr.Column():
        #     regen_btn = gr.Button("🔄 Regenerate")
        with gr.Column():
            share_btn = gr.Button("📤 Share")
    
    # Set up event handlers
    category_dropdown.change(
        update_prompt_options,
        inputs=[category_dropdown],
        outputs=[prompt_dropdown]
    )
    
    prompt_dropdown.change(
        update_reference_answer,  # 处理函数
        inputs=[prompt_dropdown],  # 输入组件
        outputs=[reference_answer]  # 输出组件
    )

    
    input_args = [prompt_dropdown, txt, chatbot_baseline, chatbot_adaptive]
    output_args = [txt, chatbot_baseline, chatbot_adaptive, latency_baseline, latency_adaptive, speedup, token_saved]
    # prompt_dropdown.change(
    #     predict,
    #     input_args,
    #     output_args
    # )

    
    # 处理文本框提交
    txt.submit(
        predict,
        input_args,
        output_args
    )


    submit_btn.click(
        predict,
        input_args,
        output_args
    )
    # prompt_dropdown.change(
    #     user_input,  # 首先清空输入框
    #     inputs=[prompt_dropdown],
    #     outputs=[txt]
    # ).then(
    #     predict,  # 然后开始流式生成
    #     inputs=[prompt_dropdown, chatbot_baseline, chatbot_adaptive],
    #     outputs=[txt, chatbot_baseline, chatbot_adaptive]
    # )

    # txt.submit(predict, [txt, chatbot_baseline, chatbot_adaptive], outputs=[txt, chatbot_baseline, chatbot_adaptive])
    
    clear_btn.click(
        lambda: ([], [], "", "", "", "", ""),
        None,
        [chatbot_baseline, chatbot_adaptive, latency_baseline, latency_adaptive, speedup, token_saved],
        queue=False
    )
    
    # Update random button to select a random prompt from a random category.
    def get_random_prompt():
        pairs = [(cat, item["prompt"], item["answer"]) for cat in PROMPT_CATEGORIES.keys() for item in PROMPT_CATEGORIES[cat]]
        cat, prompt, answer = pairs[int(time.time()) % len(pairs)]
        # update_prompt_options(cat)
        # update_reference_answer(prompt)
        answer= {"answer": answer}
        return [prompt, cat, prompt, answer]
    
    random_btn.click(
        get_random_prompt,
        None,
        [txt, category_dropdown, prompt_dropdown, reference_answer]
        # [txt]
    )

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/workspace/model/DeepSeek-R1-Distill-Qwen-7B")
    #demo.queue(concurrency_count=3)  
    demo.launch(share=True, max_threads=3)
    # demo.launch()
