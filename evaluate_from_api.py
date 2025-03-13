import os
import concurrent.futures
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

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))
NUM_THREADS=int(os.getenv("NUM_THREADS"))
random.seed(12345)

def get_client():
    if args.model_name in ["gpt-4", "gpt-4o", "o1-preview"]:
        openai.api_key = API_KEY
        client = openai
    elif args.model_name in ["deepseek-chat", "deepseek-coder"]:
        client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com/")
    elif "deepseek-r1" in args.model_name.lower():
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash-8b", "gemini-002-pro", "gemini-002-flash"]:
        genai.configure(api_key=API_KEY)
        generation_config = {
            "temperature": 0.0,
            "top_p": 1,
            "max_output_tokens": 4000,
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
        client = genai.GenerativeModel(
            model_name=args.model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
    elif args.model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
        client = anthropic.Anthropic(
            api_key=API_KEY,
        )
    elif args.model_name in ["iask"]:
        client = {"Authorization": f"Bearer {API_KEY}"}
    else:
        client = None
        print("For other model API calls, please implement the client definition method yourself.")
    return client


def call_api(client, instruction, inputs):
    start = time.time()
    if args.model_name in ["gpt-4", "gpt-4o", "deepseek-chat", "deepseek-coder"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
          model=args.model_name,
          messages=message_text,
          temperature=0,
          max_tokens=4000,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
        )
        result = completion.choices[0].message.content
    elif "deepseek-r1" in args.model_name.lower():
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
            model=args.model_name,
            messages=message_text,
            temperature=0.6,
            top_p=0.95,
            stream=True,
            max_tokens=MAX_TOKENS,
        )
        reasoning_content = ""
        content = ""
        for chunk in completion:
            # Extract the delta from choices[0]
            delta = chunk.choices[0].delta

            # If there is a reasoning_content field, add it to the accumulator.
            # (It is assumed that either "reasoning_content" or "content" will be provided per chunk.)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning_content += delta.reasoning_content
            # Otherwise, if there's a content field, add it both to the accumulator and to the file.
            elif hasattr(delta, "content") and delta.content:
                text = delta.content
                content += text
        result = content
    elif args.model_name in ["o1-preview"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
          model=args.model_name,
          messages=message_text,
        )
        result = completion.choices[0].message.content
    elif args.model_name in ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.5-flash-8b"]:
        chat_session = client.start_chat(
            history=[]
        )
        result = chat_session.send_message(instruction + inputs).text
    elif args.model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
        message = client.messages.create(
            model=args.model_name,
            max_tokens=4000,
            system="",
            messages=[
                {"role": "user", "content": instruction + inputs}
            ],
            temperature=0.0,
            top_p=1,
        )
        result = message.content[0].text
    elif args.model_name in ["iask"]:
        payload = {
            "prompt": instruction + inputs,
            "mode": "truth",
            "detail_level": "detailed",
            "stream": False
        }
        response = requests.post("https://api.iask.ai/v1/query", headers=client, json=payload, timeout=300)
        if response.status_code != 200:
            print("API call failed with status code", response.status_code, response.json())
            return response.json()["response"]["message"]
        else:
            result = response.json()["response"]["message"]
        return result
    else:
        print("For other model API calls, please implement the request method yourself.")
        result = None
    print("cost time", time.time() - start)
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


def single_request(client, single_question, cot_examples_dict, exist_result):
    exist = True
    q_id = single_question["question_id"]
    for each in exist_result:
        if q_id == each["question_id"] and single_question["question"] == each["question"]:
            pred = extract_answer(each["model_outputs"])
            return pred, each["model_outputs"], exist
    exist = False
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    # make zero-shot
    # for each in cot_examples:
    #     prompt += format_example(each["question"], each["options"], each["cot_content"])
    input_text = format_example(question, options)
    try:
        response = call_api(client, prompt, input_text)
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, None, exist
    pred = extract_answer(response)
    return pred, response, exist


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


def evaluate(subjects):
    client = get_client()
    test_df, dev_df = load_mmlu_pro()
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, subject + "_result.json")
        output_summary_path = os.path.join(args.output_dir, subject + "_summary.json")
        res, category_record = update_result(output_res_path)

        for each in tqdm(test_data):
            label = each["answer"]
            category = subject
            pred, response, exist = single_request(client, each, dev_df, res)
            if response is not None:
                res, category_record = update_result(output_res_path)
                if category not in category_record:
                    category_record[category] = {"corr": 0.0, "wrong": 0.0}
                each["pred"] = pred
                each["model_outputs"] = response
                merge_result(res, each)
                if pred is not None:
                    if pred == label:
                        category_record[category]["corr"] += 1
                    else:
                        category_record[category]["wrong"] += 1
                else:
                    category_record[category]["wrong"] += 1
                save_res(res, output_res_path)
                save_summary(category_record, output_summary_path)
                res, category_record = update_result(output_res_path)
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)


def save_res(res, output_res_path):
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
        else:
            continue
    res = temp
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))


def save_summary(category_record, output_summary_path):
    total_corr = 0.0
    total_wrong = 0.0
    for k, v in category_record.items():
        if k == "total":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"])
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    acc = total_corr / (total_corr + total_wrong)
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))


def parallel_run(subjects):
    """
    Parallel version of the evaluation. Instead of calling `single_request` one by one,
    we use ThreadPoolExecutor to dispatch multiple calls concurrently.
    """
    client = get_client()
    test_df, dev_df = load_mmlu_pro()

    # If not specified, run all subjects
    if not subjects:
        subjects = list(test_df.keys())

    print("assigned subjects", subjects)

    for subject in subjects:
        test_data = test_df[subject]

        # output files
        output_res_path = os.path.join(args.output_dir, subject + "_result.json")
        output_summary_path = os.path.join(args.output_dir, subject + "_summary.json")

        # Load existing results from disk; compute partial stats
        existing_results, category_record = update_result(output_res_path)

        # For concurrency stats, ensure subject in category_record
        if subject not in category_record:
            category_record[subject] = {"corr": 0.0, "wrong": 0.0}

        # Filter out questions that already exist in results
        questions_to_process = []
        existing_question_ids = set(
            (r["question_id"], r["question"]) for r in existing_results
        )

        for each in test_data:
            key = (each["question_id"], each["question"])
            if key not in existing_question_ids:
                questions_to_process.append(each)

        # --- Worker function for parallel calls ---
        def worker(question_item):
            # Returns (question_item, pred, response, exist)
            pred, response, exist = single_request(client, question_item, dev_df, existing_results)
            return (question_item, pred, response, exist)

        # We run the calls in parallel
        # Adjust max_workers as you see fit (watch out for rate limits)
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            # Map the worker to each question
            future_to_question = {
                executor.submit(worker, q): q for q in questions_to_process
            }

            # As results complete, update in memory
            for future in tqdm(concurrent.futures.as_completed(future_to_question),
                               total=len(questions_to_process),
                               desc=f"Processing {subject}"):
                question_item, pred, response, exist = future.result()

                # If the API returned something new, record it:
                if (response is not None) and (exist is False):
                    question_item["pred"] = pred
                    question_item["model_outputs"] = response
                    # Merge into in-memory `existing_results`
                    merge_result(existing_results, question_item)

                    # Update correct/wrong counters
                    if pred is not None and pred == question_item["answer"]:
                        category_record[subject]["corr"] += 1
                    else:
                        category_record[subject]["wrong"] += 1

        # All parallel calls are done; now save final results
        save_res(existing_results, output_res_path)
        save_summary(category_record, output_summary_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4")
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    assigned_subjects = []
    args = parser.parse_args()

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    os.makedirs(args.output_dir, exist_ok=True)
    parallel_run(assigned_subjects)
    # evaluate(assigned_subjects)
