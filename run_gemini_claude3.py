import os
import google.generativeai as genai
import datasets
import json
from typing import List
import re
import time
import tqdm
import multiprocessing
import random
import argparse
from datetime import datetime
import anthropic

now = datetime.now()
current_time = now.strftime("%H_%M_%S")

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='pro', type=str)
parser.add_argument("--thread", default=40, type=int)
parser.add_argument("--shots", default=0, type=int)
parser.add_argument("--prompt", default='gdm', type=str)
parser.add_argument("--tiny", action='store_true', default=False)
args = parser.parse_args()


if args.model in ['pro', 'flash']:
  print('setting up Gemini evaluation')
  genai.configure(api_key=os.environ["GEMINI_API_KEY"])
  generation_config = {
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1000,
    "response_mime_type": "text/plain",
  }
  safety_settings = [
    {
      "category": "HARM_CATEGORY_HARASSMENT",
      "threshold": "BLOCK_NONE",
    },
    {
      "category": "HARM_CATEGORY_HATE_SPEECH",
      "threshold": "BLOCK_NONE",
    },
    {
      "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
      "threshold": "BLOCK_NONE",
    },
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "threshold": "BLOCK_NONE",
    },
  ]
  model = genai.GenerativeModel(
    model_name=f"gemini-1.5-{args.model}-latest",
    safety_settings=safety_settings,
    generation_config=generation_config,
  )
elif args.model in ['opus', 'sonnet']:
  print('setting up Claude3 evaluation')
  client = anthropic.Anthropic(
      api_key=os.environ["CLAUDE_API_KEY"],
  )
else:
  raise NotImplementedError(args.model)


def form_options(options: list):
  option_str = 'Options are:\n'
  opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  for opt, o in zip(options, opts):
      option_str += f'({o}): {opt}' + '\n'
  return option_str


def get_prediction(output):
  #pattern = r'answer is \(?([ABCDEFGHIJ])\)?'
  pattern = r"(Answer:|answer is)\s*\(?([ABCDEFGHIJ])\)?"
  match = re.search(pattern, output)
  if match:
    return match.group(2)
  else:
    pattern = r' (:|is)\s*\(?([ABCDEFGHIJ])\)?\b'
    match = re.search(pattern, output)
    if match:
      return match.group(2)
    else:
      print('extraction failed, do a random guess')
      return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])


def run_one_question(question: str):
  count = 0
  max_trials = 10
  if args.model in ['pro', 'flash']:
    chat_session = model.start_chat(
      history=[]
    )
    while True:
      try:
        response = chat_session.send_message(question)
        break
      except Exception as e:
        if 'quota' in str(e):
          count += 1
          if count >= max_trials:
            break
          else:
            time.sleep(10)
        else:
          return f'Exception: {e}'

  elif args.model in ['opus', 'sonnet']:
    while True:
      try:
        message = client.messages.create(
            model=f"claude-3-{args.model}-20240229",
            max_tokens=1024,
            system="",
            messages=[
                {"role": "user", "content": question}
            ],
            temperature = 0.0,
            top_p = 1,
        )
        response = message.content[0]
        break
      except Exception:
        count += 1
        if count >= max_trials:
          break
        else:
          time.sleep(5)
  else:
    raise NotImplementedError(args.model)
  
  try:
    return response.text
  except Exception:
    print('failed to give results!')
    return 'The answer is (A)'

categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
              'health', 'physics', 'business', 'philosophy', 'economics', 'other',
              'psychology', 'history']

prompts = {}
for c in categories:
  prompts[c] = ''
  count = 0
  for d in datasets.load_dataset('TIGER-Lab/MMLU-Pro', split='validation'):
    if d['category'] == c and count < args.shots:
      prompts[d['category']] += d['question'] + '\n' + form_options(d['options']) + '\n' + d['cot_content'] + '\n\n'
      count += 1


def func(line):
  if args.prompt == 'gdm':
    system_prompt = "Finish your answer with Final Answer: (X) where X is the correct letter choice. If none or more than one of the options match, choose the one that is the closest.\n\n"
  else:
    system_prompt = "Finish your answer with the answer is (X) where X is the correct letter choice. If none or more than one of the options match, choose the one that is the closest.\n\n"

  prefix = prompts[line['category']]
  query = system_prompt + prefix + line['question'] + '\n' + form_options(line['options']) + '\n'

  solution = run_one_question(query)
  solution = solution.replace('**', '')

  pred = get_prediction(solution)

  line['rationale'] = solution
  line['pred'] = pred
  return line


if __name__ == "__main__":
  per_category_accuracy = {c: [0, 0] for c in categories}

  print(f'----------------- Start Answering -------------------')
  dataset = datasets.load_dataset('TIGER-Lab/MMLU-Pro', split='test')
  dataset = dataset.to_list()

  if args.tiny:
    dataset = dataset[:400]

  if args.thread == 1:
    results = []
    for d in tqdm.tqdm(dataset):
      try:
        results.append(func(d))
      except Exception:
        continue
  elif args.thread > 1:
    with multiprocessing.Pool(args.thread) as p:
      results = list(tqdm.tqdm(p.imap(func, dataset), total=len(dataset)))
  else:
    raise ValueError(args.thread)

  success, fail = 0, 0
  for line in results:
    if line['pred'] == line['answer']:
      success += 1
      per_category_accuracy[line['category']][0] += 1
    else:
      fail += 1
      per_category_accuracy[line['category']][1] += 1

  print(success / (success + fail))

  with open(f'model_outputs_{args.model}_{args.shots}shots_{current_time}.json', 'w') as f:
    json.dump(results, f, indent=2)

  for k, v in per_category_accuracy.items():
    print('accuracy: ', k, v[0] / (v[0] + v[1]))
