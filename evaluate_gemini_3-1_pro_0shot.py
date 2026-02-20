#!/usr/bin/env python3
"""
Evaluate gemini-3.1-pro-preview on MMLU-Pro benchmark (0-shot).
Supports multi-process sharding and resume.

Usage:
    # Single process:
    python evaluate_gemini_3-1_pro.py --output_dir eval_results

    # Multi-process (e.g., 4 shards, run in separate terminals):
    python evaluate_gemini_3-1_pro.py -o eval_results --num_shards 4 --shard_id 0
    python evaluate_gemini_3-1_pro.py -o eval_results --num_shards 4 --shard_id 1
    python evaluate_gemini_3-1_pro.py -o eval_results --num_shards 4 --shard_id 2
    python evaluate_gemini_3-1_pro.py -o eval_results --num_shards 4 --shard_id 3

    # Merge all shard results and print summary:
    python evaluate_gemini_3-1_pro.py -o eval_results --num_shards 20 --merge
"""

import os
import json
import re
import random
import time
import argparse
import logging
from tqdm import tqdm
from datasets import load_dataset
from google import genai

random.seed(12345)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.1-pro-preview"
MAX_RETRIES = 5


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
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 4000,
            },
        )
        if response.text is None:
            raise RuntimeError(
                f"Empty response. Finish reason: {response.candidates[0].finish_reason}"
            )
        return response.text


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df = preprocess(dataset["test"])
    return test_df


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


# ─── Prompt Formatting (0-shot) ─────────────────────────────────────────────

def format_example(question, options):
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    example += "Answer: "
    return example


def build_prompt(question_data):
    category = question_data["category"]
    prompt = (
        f"The following is a multiple choice question (with answer) about "
        f"{category}. Think step by step and then output the answer in the "
        f"format of \"The answer is (X)\" at the end.\n\n"
    )
    prompt += format_example(question_data["question"], question_data["options"])
    return prompt


# ─── Answer Extraction (reused) ─────────────────────────────────────────────

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


# ─── Result I/O (reused) ────────────────────────────────────────────────────

def load_result(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load {path}: {e}")
    return []


def save_res(res, path):
    """Save results with deduplication by question_id."""
    seen = set()
    deduped = []
    for each in res:
        qid = each["question_id"]
        if qid not in seen:
            seen.add(qid)
            deduped.append(each)
    with open(path, "w") as f:
        json.dump(deduped, f, ensure_ascii=False)


def compute_summary(results):
    """Compute per-category and overall accuracy."""
    rng = random.Random(12345)  # isolated RNG for reproducible random guessing
    category_record = {}
    for each in results:
        cat = each["category"]
        if cat not in category_record:
            category_record[cat] = {"corr": 0.0, "wrong": 0.0}
        pred = each.get("pred")
        if not pred:
            # Random guess for failed extractions
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
    logger.info("Loading MMLU-Pro dataset...")
    test_df = load_mmlu_pro()

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

    # Shard assignment: each process handles questions where index % num_shards == shard_id
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

    # Load existing results for resume
    processed_ids = set()
    results_by_subject = {}
    for subject in subjects:
        res_path = os.path.join(shard_dir, f"{subject}_result.json")
        existing = load_result(res_path)
        results_by_subject[subject] = existing
        for r in existing:
            processed_ids.add(r["question_id"])

    skipped = sum(1 for q in shard_questions if q["question_id"] in processed_ids)
    logger.info(f"Resume: {skipped} already processed, {len(shard_questions) - skipped} remaining")

    # Process questions
    new_count = 0
    for q in tqdm(shard_questions, desc=f"Shard {args.shard_id}"):
        qid = q["question_id"]
        subject = q["category"]

        # Skip already processed (resume)
        if qid in processed_ids:
            continue

        # Build 0-shot prompt (no few-shot examples)
        prompt = build_prompt(q)

        # Call API with exponential backoff
        response = None
        for attempt in range(MAX_RETRIES):
            try:
                start = time.time()
                response = client.get_response(prompt)
                response = response.replace("**", "")
                elapsed = time.time() - start
                logger.debug(f"API call for {qid} took {elapsed:.2f}s")
                break
            except Exception as e:
                wait = min(2 ** attempt, 60)
                logger.warning(
                    f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        if response is None:
            logger.error(f"Failed after {MAX_RETRIES} retries for question {qid}, skipping.")
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

        # Save after each question for robustness
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
        f"Processed {new_count} new questions. "
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

    # Deduplicate and collect for summary
    all_merged = []
    for subject in sorted(all_results_by_subject.keys()):
        res = all_results_by_subject[subject]
        save_res(res, os.path.join(merged_dir, f"{subject}_result.json"))
        seen = set()
        for r in res:
            if r["question_id"] not in seen:
                seen.add(r["question_id"])
                all_merged.append(r)

    summary = compute_summary(all_merged)
    save_summary(summary, os.path.join(merged_dir, "summary.json"))

    # Print results table
    total = summary["total"]
    n_total = int(total["corr"] + total["wrong"])
    print(f"\n{'=' * 65}")
    print(f"  MMLU-Pro Results (0-shot)  —  {args.model}")
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
        description="Evaluate gemini-3.1-pro-preview on MMLU-Pro (0-shot)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="eval_results/",
        help="Base output directory for results",
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
    args = parser.parse_args()

    if args.merge:
        merge_shards(args)
    else:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            parser.error(
                f"shard_id ({args.shard_id}) must be in [0, {args.num_shards})"
            )
        evaluate(args)