import csv
import json
import argparse
import os
import torch
import numpy as np
import random
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForCausalLM
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
from distutils.util import strtobool
import logging
import sys
from datasets import load_dataset

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 2048
max_new_tokens = 256


def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def load_model():
    try:
        llm = LLM(model=args.model, gpu_memory_utilization=float(args.gpu_util),
                  tensor_parallel_size=args.ngpu, max_model_len=4096,
                  trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0, max_tokens=256,
                                         stop=["Question:"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print("vllm unsupported models", e)
        return None, None
    return (llm, sampling_params), tokenizer


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
    return res_df


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        prompt += example["cot_content"] + "\n\n"
    else:
        prompt += "A: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def check_exist(res, q_id):
    for each in res:
        if q_id == each["question_id"]:
            if "pred" in each:
                # logging.debug("exist, skip it")
                return True
            else:
                logging.debug("no result in exist result error")
                return False
        else:
            continue
    return False


def extract_answer(text):
    pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        logging.info("answer extract failed\n" + text)
        return None


def batch_inference(llm, sampling_params, inference_batch):
    start = time.time()
    outputs = llm.generate(inference_batch, sampling_params)
    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
                # print("random hit.")
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path, exists_result=None):
    llm, sampling_params = model
    if not exists_result:
        res = []
    else:
        res = exists_result
    print("load exists result length", len(res))
    global choices
    logging.info("evaluating " + subject)
    batch_size = args.batch_size
    inference_batches = []
    label_batches = []
    in_batch_index = []

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
        options_num = len(test_df[i]["options"])
        if options_num != 10 and options_num != 4:
            print("options_num", options_num)
        curr = test_df[i]
        q_id = curr["question_id"]
        if check_exist(res, q_id):
            continue
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)
        in_batch_index.append(i)

    i = 0
    while i < len(test_df):
        if i + batch_size < len(test_df):
            end_index = i + batch_size
        else:
            end_index = len(test_df)
        curr_batch = inference_batches[i: end_index]
        pred_batch, response_batch = batch_inference(llm, sampling_params, curr_batch)
        index_list = in_batch_index[i: end_index]
        for j, index in enumerate(index_list):
            curr = test_df[index]
            curr["pred"] = pred_batch[j]
            curr["generated_text"] = response_batch[j]
            res.append(curr)
        accu, corr, wrong = save_res(res, output_path)
        logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))
        i += batch_size
    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong


def main():
    model, tokenizer = load_model()
    model.eval()
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    full_test_df, full_val_df = load_mmlu_pro()
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)
    logging.info("selected subjects:\n" + "\n".join(selected_subjects))
    print("selected subjects:\n" + "\n".join(selected_subjects))
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        output_path = os.path.join(save_result_dir, "{}".format(subject))
        if os.path.exists(output_path):
            with open(output_path, "r") as fi:
                exists_result = json.load(fi)
        else:
            exists_result = []
        acc, corr_count, wrong_count = eval_cot(subject, model, tokenizer, val_df,
                                                test_df, output_path, exists_result)
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--ngpu", "-g", type=int, default=1)
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    main()

'''
model="/ML-A100/team/mm/zhangge/Llama-2-7b-hf"
model="google/gemma-7b"
model="/ML-A100/team/mm/zhangge/Llama-2-13b-hf"
model="/mnt/tjena/shared/Meta-Llama-3-8B"
model="/ML-A100/team/mm/zhangge/Llama-2-7b-chat-hf"
model="01-ai/Yi-6B"
model="mistralai/Mixtral-8x7B-v0.1"
meta-llama
Llama-2-7b-hf

variance
ranking big difference
'''

