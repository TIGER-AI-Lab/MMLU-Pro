#!/usr/bin/env python3
"""
Evaluate gemini-3.1-pro-preview on MMLU-Pro benchmark.
Supports multi-process sharding and resume.

Usage:
    # Download dataset to local first:
    python evaluate_gemini_3-1_pro.py --download --data_dir mmlu_pro_data

    # Single process:
    python evaluate_gemini_3-1_pro.py --data_dir mmlu_pro_data --output_dir eval_results

    # Multi-process (e.g., 4 shards, run in separate terminals):
    python evaluate_gemini_3-1_pro.py --data_dir mmlu_pro_data -o eval_results --num_shards 4 --shard_id 0
    python evaluate_gemini_3-1_pro.py --data_dir mmlu_pro_data -o eval_results --num_shards 4 --shard_id 1
    python evaluate_gemini_3-1_pro.py --data_dir mmlu_pro_data -o eval_results --num_shards 4 --shard_id 2
    python evaluate_gemini_3-1_pro.py --data_dir mmlu_pro_data -o eval_results --num_shards 4 --shard_id 3

    # Merge all shard results and print summary:
    python evaluate_gemini_3-1_pro.py --data_dir mmlu_pro_data -o eval_results --num_shards 20 --merge
"""

import os
import json
import re
import random
import time
import argparse
import logging
from tqdm import tqdm
from google import genai

random.seed(12345)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-pro-preview"
MAX_RETRIES = 10
MAX_FORMAT_RETRIES = 10  # 针对未匹配到 "The answer is" 的重试次数


# ─── Gemini Client ───────────────────────────────────────────────────────────

class GeminiClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self.client = genai.Client()
        self.model = model

    def get_response(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": 1.0,
                "top_p": 0.95,
                "max_output_tokens": 32000,
            },
        )
        if response.text is None:
            raise RuntimeError(
                f"Empty response. Finish reason: {response.candidates[0].finish_reason}"
            )
        return response.text


# ─── Data Download & Local Loading ───────────────────────────────────────────

def download_mmlu_pro(data_dir: str):
    """Download MMLU-Pro from HuggingFace and save as local JSON files."""
    from datasets import load_dataset

    os.makedirs(data_dir, exist_ok=True)
    logger.info("Downloading MMLU-Pro dataset from HuggingFace...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")

    test_path = os.path.join(data_dir, "test.json")
    val_path = os.path.join(data_dir, "validation.json")

    test_data = [dict(item) for item in dataset["test"]]
    val_data = [dict(item) for item in dataset["validation"]]

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False)
    logger.info(f"Saved {len(test_data)} test samples to {test_path}")

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False)
    logger.info(f"Saved {len(val_data)} validation samples to {val_path}")

    logger.info("Download complete.")


def load_mmlu_pro_local(data_dir: str):
    """Load MMLU-Pro from local JSON files."""
    test_path = os.path.join(data_dir, "test.json")
    val_path = os.path.join(data_dir, "validation.json")

    if not os.path.exists(test_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Local data not found in '{data_dir}'. "
            f"Please run with --download first to download the dataset."
        )

    logger.info(f"Loading test data from {test_path}...")
    with open(test_path, "r", encoding="utf-8") as f:
        test_raw = json.load(f)

    logger.info(f"Loading validation data from {val_path}...")
    with open(val_path, "r", encoding="utf-8") as f:
        val_raw = json.load(f)

    test_df = preprocess(test_raw)
    val_df = preprocess(val_raw)

    logger.info(
        f"Loaded {sum(len(v) for v in test_df.values())} test questions, "
        f"{sum(len(v) for v in val_df.values())} validation questions"
    )
    return test_df, val_df


def preprocess(df):
    res_df = []
    for each in df:
        each = dict(each)  # make mutable copy
        options = [opt for opt in each["options"] if opt != "N/A"]
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        res.setdefault(each["category"], []).append(each)
    return res


# ─── Prompt Formatting ──────────────────────────────────────────────────────

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


def build_prompt(question_data, cot_examples):
    category = question_data["category"]
    prompt = (
        f"The following are multiple choice questions (with answers) about "
        f"{category}. Think step by step and then output the answer in the "
        f"format of \"The answer is (X)\" at the end.\n\n"
    )
    for ex in cot_examples:
        prompt += format_example(ex["question"], ex["options"], ex["cot_content"])
    prompt += format_example(question_data["question"], question_data["options"])
    return prompt


def build_retry_prompt(question_data, cot_examples, previous_response):
    """构建重试 prompt，强调必须以 'The answer is (X)' 格式结尾。"""
    category = question_data["category"]
    prompt = (
        f"The following are multiple choice questions (with answers) about "
        f"{category}. Think step by step and then output the answer in the "
        f"format of \"The answer is (X)\" at the end.\n\n"
        f"**IMPORTANT: You MUST end your response with exactly "
        f"\"The answer is (X)\" where X is one of the option letters "
        f"(A, B, C, D, etc.). This is mandatory. Do NOT omit this.**\n\n"
    )
    for ex in cot_examples:
        prompt += format_example(ex["question"], ex["options"], ex["cot_content"])
    prompt += format_example(question_data["question"], question_data["options"])

    prompt += (
        f"\n\n[Your previous response did not end with \"The answer is (X)\". "
        f"Here was your previous response for reference:]\n"
        f"{previous_response}\n\n"
        f"[Please re-answer the question above. You MUST conclude your response "
        f"with \"The answer is (X)\" where X is the correct option letter.]\n"
        f"Answer: "
    )
    return prompt


# ─── Answer Extraction ──────────────────────────────────────────────────────

def has_answer_is_pattern(text: str) -> bool:
    """检查回复中是否包含 'The answer is (X)' 模式。支持前面有换行等任意字符。"""
    return bool(re.search(r"[Tt]he answer is \(?[A-J]\)?", text))


def extract_answer(text):
    # Level 1: "answer is (X)" or "answer is X"
    match = re.search(r"answer is \(?([A-J])\)?", text)
    if match:
        return match.group(1)
    # Level 2: "Answer: X"
    match = re.search(r"[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    # Level 3: last standalone capital letter A-J in text
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


# ─── API Call with Format Retry ─────────────────────────────────────────────

def call_api_with_retries(client, prompt, qid):
    """调用 API，带指数退避重试。"""
    for attempt in range(MAX_RETRIES):
        try:
            start = time.time()
            response = client.get_response(prompt)
            response = response.replace("**", "")
            elapsed = time.time() - start
            logger.debug(f"API call for {qid} took {elapsed:.2f}s")
            return response
        except Exception as e:
            wait = min(2 ** attempt, 60)
            logger.warning(
                f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)
    return None


def get_response_with_format_retry(client, question_data, cot_examples, qid,
                                   previous_response=None):
    """
    调用模型获取回复。如果回复中没有匹配到 'The answer is (X)' 的模式，
    则使用强调格式的 prompt 进行重试，最多重试 MAX_FORMAT_RETRIES 次。

    如果提供了 previous_response（例如从 resume 中读取的旧结果），
    则直接从格式重试开始，跳过首次正常调用。
    """
    if previous_response is not None:
        # 直接从格式重试开始（resume 场景）
        response = previous_response
        logger.info(f"Question {qid}: resuming with format retry (previous output lacks pattern)")
    else:
        # 第一次正常调用
        prompt = build_prompt(question_data, cot_examples)
        response = call_api_with_retries(client, prompt, qid)

        if response is None:
            return None

        if has_answer_is_pattern(response):
            return response

    # 没有匹配到，进行格式重试
    for retry in range(MAX_FORMAT_RETRIES):
        logger.warning(
            f"Question {qid}: response missing 'The answer is (X)' pattern. "
            f"Format retry {retry + 1}/{MAX_FORMAT_RETRIES}..."
        )
        retry_prompt = build_retry_prompt(
            question_data, cot_examples, response
        )
        response = call_api_with_retries(client, retry_prompt, qid)

        if response is None:
            return None

        if has_answer_is_pattern(response):
            logger.info(
                f"Question {qid}: format retry {retry + 1} succeeded."
            )
            return response

    logger.warning(
        f"Question {qid}: all {MAX_FORMAT_RETRIES} format retries exhausted. "
        f"Using last response for fallback extraction."
    )
    return response


# ─── Result I/O ─────────────────────────────────────────────────────────────

def load_result(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load {path}: {e}")
    return []


def save_res(res, path):
    """Save results with deduplication by question_id. Keeps the LAST occurrence
    so that re-processed results overwrite old ones."""
    seen = {}
    for each in res:
        qid = each["question_id"]
        seen[qid] = each  # later entries overwrite earlier ones
    deduped = list(seen.values())
    with open(path, "w") as f:
        json.dump(deduped, f, ensure_ascii=False)


def compute_summary(results):
    """Compute per-category and overall accuracy."""
    rng = random.Random(12345)
    category_record = {}
    for each in results:
        cat = each["category"]
        if cat not in category_record:
            category_record[cat] = {"corr": 0.0, "wrong": 0.0}
        pred = each.get("pred")
        if not pred:
            x = rng.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                category_record[cat]["corr"] += 1
            else:
                category_record[cat]["wrong"] += 1
        elif pred == each["answer"]:
            category_record[cat]["corr"] += 1
        else:
            category_record[cat]["wrong"] += 1

    total_corr = sum(v["corr"] for v in category_record.values())
    total_wrong = sum(v["wrong"] for v in category_record.values())
    for v in category_record.values():
        total = v["corr"] + v["wrong"]
        v["acc"] = v["corr"] / total if total > 0 else 0.0
    overall = total_corr + total_wrong
    category_record["total"] = {
        "corr": total_corr,
        "wrong": total_wrong,
        "acc": total_corr / overall if overall > 0 else 0.0,
    }
    return category_record


def save_summary(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# ─── Evaluation ─────────────────────────────────────────────────────────────

def get_shard_dir(args):
    if args.num_shards > 1:
        return os.path.join(args.output_dir, f"shard_{args.shard_id}")
    return args.output_dir


def evaluate(args):
    client = GeminiClient(model=args.model)
    logger.info(f"Model: {args.model}")
    logger.info(f"Loading MMLU-Pro dataset from local directory: {args.data_dir}")
    test_df, dev_df = load_mmlu_pro_local(args.data_dir)

    # Determine subjects
    if args.assigned_subjects == "all":
        subjects = sorted(test_df.keys())
    else:
        subjects = [s.strip() for s in args.assigned_subjects.split(",")]

    # Flatten all questions across selected subjects
    all_questions = []
    for subject in subjects:
        if subject not in test_df:
            logger.warning(f"Subject '{subject}' not found in dataset, skipping.")
            continue
        for q in test_df[subject]:
            all_questions.append(q)
    logger.info(f"Total questions: {len(all_questions)} across {len(subjects)} subjects")

    # Shard assignment
    shard_questions = [
        q for i, q in enumerate(all_questions)
        if i % args.num_shards == args.shard_id
    ]
    logger.info(
        f"Shard {args.shard_id}/{args.num_shards}: "
        f"{len(shard_questions)} questions assigned"
    )

    # Output directory
    shard_dir = get_shard_dir(args)
    os.makedirs(shard_dir, exist_ok=True)

    # ── Load existing results for resume ──
    # 分为两类：
    #   1. 已有结果且 model_outputs 包含 "The answer is" → 跳过
    #   2. 已有结果但 model_outputs 不包含 "The answer is" → 需要重试
    processed_ids = set()          # 已完成且格式正确，可跳过
    needs_format_retry = {}        # qid -> old model_outputs，需要重新处理
    results_by_subject = {}

    for subject in subjects:
        res_path = os.path.join(shard_dir, f"{subject}_result.json")
        existing = load_result(res_path)
        kept = []
        for r in existing:
            qid = r["question_id"]
            model_output = r.get("model_outputs", "")
            if has_answer_is_pattern(model_output):
                # 格式正确，保留并标记为已处理
                processed_ids.add(qid)
                kept.append(r)
            else:
                # 格式不正确，记录旧输出用于重试，不加入 processed_ids
                needs_format_retry[qid] = model_output
                logger.info(
                    f"Question {qid} ({subject}): previous output lacks "
                    f"'The answer is (X)' pattern, will re-process."
                )
                # 不加入 kept，后面重新处理后会追加新结果
        results_by_subject[subject] = kept

    skipped = sum(1 for q in shard_questions if q["question_id"] in processed_ids)
    retry_count = sum(
        1 for q in shard_questions if q["question_id"] in needs_format_retry
    )
    remaining = len(shard_questions) - skipped
    logger.info(
        f"Resume: {skipped} already done (format OK), "
        f"{retry_count} need format retry, "
        f"{remaining - retry_count} new. "
        f"Total to process: {remaining}"
    )

    # Process questions
    new_count = 0

    for q in tqdm(shard_questions, desc=f"Shard {args.shard_id}"):
        qid = q["question_id"]
        subject = q["category"]

        # Skip already processed with correct format
        if qid in processed_ids:
            continue

        cot_examples = dev_df.get(subject, [])

        # 判断是全新的还是需要格式重试的
        previous_output = needs_format_retry.get(qid, None)

        response = get_response_with_format_retry(
            client, q, cot_examples, qid,
            previous_response=previous_output,
        )

        if response is None:
            logger.error(f"Failed after all retries for question {qid}, skipping.")
            continue

        pred = extract_answer(response)
        if pred is None:
            logger.warning(f"Answer extraction failed for question {qid}")

        # Build result entry
        result_entry = dict(q)
        result_entry["pred"] = pred
        result_entry["model_outputs"] = response

        results_by_subject.setdefault(subject, []).append(result_entry)
        processed_ids.add(qid)
        new_count += 1

        # Save after each question (save_res keeps last occurrence → overwrites old)
        save_res(
            results_by_subject[subject],
            os.path.join(shard_dir, f"{subject}_result.json"),
        )

    # Final save for all subjects
    for subject in subjects:
        if results_by_subject.get(subject):
            save_res(
                results_by_subject[subject],
                os.path.join(shard_dir, f"{subject}_result.json"),
            )

    # Compute and save shard summary
    all_results = []
    for subj_results in results_by_subject.values():
        all_results.extend(subj_results)
    summary = compute_summary(all_results)
    save_summary(summary, os.path.join(shard_dir, "summary.json"))

    total = summary["total"]
    n = int(total["corr"] + total["wrong"])
    logger.info(
        f"Shard {args.shard_id} complete. "
        f"Processed {new_count} new/retried questions. "
        f"Accuracy: {total['acc']:.4f} ({int(total['corr'])}/{n})"
    )


# ─── Merge Shards ───────────────────────────────────────────────────────────

def merge_shards(args):
    logger.info(f"Merging {args.num_shards} shards from {args.output_dir}")
    all_results_by_subject = {}

    for shard_id in range(args.num_shards):
        shard_dir = os.path.join(args.output_dir, f"shard_{shard_id}")
        if not os.path.exists(shard_dir):
            logger.warning(f"Shard directory {shard_dir} not found, skipping.")
            continue
        for fname in sorted(os.listdir(shard_dir)):
            if fname.endswith("_result.json"):
                subject = fname[: -len("_result.json")]
                results = load_result(os.path.join(shard_dir, fname))
                all_results_by_subject.setdefault(subject, []).extend(results)
                logger.info(
                    f"  Loaded {len(results)} results for '{subject}' from shard {shard_id}"
                )

    merged_dir = os.path.join(args.output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    all_merged = []
    for subject in sorted(all_results_by_subject.keys()):
        res = all_results_by_subject[subject]
        save_res(res, os.path.join(merged_dir, f"{subject}_result.json"))
        # Deduplicate for summary (save_res keeps last, replicate here)
        seen = {}
        for r in res:
            seen[r["question_id"]] = r
        all_merged.extend(seen.values())

    summary = compute_summary(all_merged)
    save_summary(summary, os.path.join(merged_dir, "summary.json"))

    # Print results table
    total = summary["total"]
    n_total = int(total["corr"] + total["wrong"])
    print(f"\n{'=' * 65}")
    print(f"  MMLU-Pro Results  —  {args.model}")
    print(f"{'=' * 65}")
    for k in sorted(summary.keys()):
        if k == "total":
            continue
        v = summary[k]
        n = int(v["corr"] + v["wrong"])
        bar = "█" * int(v["acc"] * 20) + "░" * (20 - int(v["acc"] * 20))
        print(f"  {k:35s} {bar} {v['acc']:.4f}  ({int(v['corr']):>4d}/{n})")
    print(f"{'─' * 65}")
    bar = "█" * int(total["acc"] * 20) + "░" * (20 - int(total["acc"] * 20))
    print(f"  {'TOTAL':35s} {bar} {total['acc']:.4f}  ({int(total['corr']):>4d}/{n_total})")
    print(f"{'=' * 65}")

    logger.info(f"Merged results saved to {merged_dir}")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate gemini-3.1-pro-preview on MMLU-Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="eval_results/",
        help="Base output directory for results",
    )
    parser.add_argument(
        "--data_dir", "-d", type=str, default="mmlu_pro_data/",
        help="Local directory for MMLU-Pro data (default: mmlu_pro_data/)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--assigned_subjects", "-a", type=str, default="all",
        help="Comma-separated subjects or 'all'",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help="Total number of parallel shards (default: 1)",
    )
    parser.add_argument(
        "--shard_id", type=int, default=0,
        help="This process's shard ID, 0-indexed (default: 0)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge results from all shards instead of evaluating",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download MMLU-Pro dataset to local data_dir and exit",
    )
    args = parser.parse_args()

    if args.download:
        download_mmlu_pro(args.data_dir)
    elif args.merge:
        merge_shards(args)
    else:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            parser.error(
                f"shard_id ({args.shard_id}) must be in [0, {args.num_shards})"
            )
        evaluate(args)