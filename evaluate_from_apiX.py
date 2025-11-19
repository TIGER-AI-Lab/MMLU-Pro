import os
import openai
from openai import OpenAI
import anthropic
import google.generativeai as genai
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse
import requests
from ai21 import AI21Client
from ai21.models.chat import ChatMessage, ResponseFormat, DocumentSchema, FunctionToolDefinition
from ai21.models.chat import ToolDefinition, ToolParameters
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

API_KEY = ""
random.seed(12345)

# Add lock for thread-safe file operations
file_lock = threading.Lock()

def get_client():
    client = OpenAI(api_key=API_KEY, base_url=args.url)
    return client

def call_api(client, instruction, inputs):
    start = time.time()
    message_text = [{"role": "user", "content": instruction + inputs}]
    completion = client.chat.completions.create(
      model=args.model_name,
      messages=message_text,
      temperature=0,
      max_tokens=32768,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
    )
    result = completion.choices[0].message.content
    print("request time", time.time() - start, "sec")
    return result

def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df

def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res

def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example

def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def single_request(client, single_question, cot_examples_dict):
    """Modified: removed exist_result parameter, check moved outside"""
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    for each in cot_examples:
        prompt += format_example(each["question"], each["options"], each["cot_content"])
    input_text = format_example(question, options)
    try:
        response = call_api(client, prompt, input_text)
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, None
    pred = extract_answer(response)
    return pred, response

def update_result(output_res_path):
    category_record = {}
    res = []
    success = False
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                    for each in res:
                        category = each["category"]
                        if category not in category_record:
                            category_record[category] = {"corr": 0.0, "wrong": 0.0}
                        if not each["pred"]:
                            x = random.randint(0, len(each["options"]) - 1)
                            if x == each["answer_index"]:
                                category_record[category]["corr"] += 1
                            else:
                                category_record[category]["wrong"] += 1
                        elif each["pred"] == each["answer"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record

def merge_result(res, curr):
    merged = False
    for i, single in enumerate(res):
        if single["question_id"] == curr["question_id"] and single["question"] == curr["question"]:
            res[i] = curr
            merged = True
    if not merged:
        res.append(curr)
    return res

def process_single_question(args_tuple):
    """New function for parallel processing"""
    client, each, dev_df, output_res_path, subject = args_tuple
    label = each["answer"]

    # Check existence inside thread
    res, category_record = update_result(output_res_path)
    q_id = each["question_id"]
    for existing in res:
        if q_id == existing["question_id"] and each["question"] == existing["question"]:
            return None  # Already processed

    pred, response = single_request(client, each, dev_df)

    with file_lock:  # Thread-safe file operations
        res, category_record = update_result(output_res_path)
        if subject not in category_record:
            category_record[subject] = {"corr": 0.0, "wrong": 0.0}

        each["pred"] = pred
        each["model_outputs"] = response

        if pred is not None:
            if pred == label:
                category_record[subject]["corr"] += 1
            else:
                category_record[subject]["wrong"] += 1
        else:
            category_record[subject]["wrong"] += 1

        res = merge_result(res, each)
        save_res(res, output_res_path)
        save_summary(category_record, os.path.join(args.output_dir, subject + "_summary.json"))

    return each

def evaluate(subjects):
    client = get_client()
    test_df, dev_df = load_mmlu_pro()
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)

    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, subject + "_result.json")

        # Prepare arguments for parallel processing
        process_args = [
            (client, each, dev_df, output_res_path, subject)
            for each in test_data
        ]

        # Process X queries at a time
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(
                executor.map(process_single_question, process_args),
                total=len(test_data),
                desc=f"Processing {subject}"
            ))

def save_res(res, output_res_path):
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
    res = temp
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))

def save_summary(category_record, output_summary_path):
    total_corr = 0.0
    total_wrong = 0.0
    for k, v in category_record.items():
        if k == "total":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"]) if (v["corr"] + v["wrong"]) > 0 else 0
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    acc = total_corr / (total_corr + total_wrong) if (total_corr + total_wrong) > 0 else 0
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--model_name", "-m", type=str, default="local")
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    parser.add_argument("--url", "-u", type=str, default="http://127.0.0.1:8080/")
    parser.add_argument("--num_workers", "-n", type=int, default=4,
                       help="Number of concurrent queries")
    assigned_subjects = []
    args = parser.parse_args()
    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(assigned_subjects)