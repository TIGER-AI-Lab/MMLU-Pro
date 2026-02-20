from collections import defaultdict
import json
import re
from typing import Dict, List, Optional
from datasets import load_dataset, concatenate_datasets

LETTER_PREFIX_RE = re.compile(r"^[A-Za-z]\s*[\.\)\:\-]\s*")
NUMBER_PREFIX_RE = re.compile(r"^\(?\d+\)?\s*[\.\)\:\-]\s*")


def normalize_option(option: Optional[str]) -> Optional[str]:
    """Return a normalized option string for comparison, stripping enumeration prefixes."""
    if option is None:
        return None

    text = option.strip()
    if not text:
        return None

    for pattern in (LETTER_PREFIX_RE, NUMBER_PREFIX_RE):
        if pattern.match(text):
            text = pattern.sub("", text, count=1)
            break

    text = re.sub(r"\s+", " ", text).strip().lower()
    return text or None


def prepare_entries(data: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for idx, item in enumerate(data):
        question = (item.get("question") or "").strip()
        if not question:
            continue

        raw_options = [opt for opt in item.get("options", []) if opt]
        normalized_options = {
            normalized
            for opt in raw_options
            if (normalized := normalize_option(opt))
        }

        entry = {
            "index": idx,
            "question": question,
            "question_id": item.get("question_id"),
            "raw_options": raw_options,
            "normalized_options": normalized_options,
        }
        grouped[question].append(entry)

    return grouped


def find_supersets(data: List[Dict]) -> List[Dict]:
    grouped_entries = prepare_entries(data)
    results_by_superset: Dict[int, Dict] = {}

    for question, entries in grouped_entries.items():
        for superset_entry in entries:
            superset_norm = superset_entry["normalized_options"]
            if len(superset_norm) < 2:
                continue

            for subset_entry in entries:
                if subset_entry["index"] == superset_entry["index"]:
                    continue

                subset_norm = subset_entry["normalized_options"]
                if not subset_norm:
                    continue

                if subset_norm < superset_norm:
                    result = results_by_superset.setdefault(
                        superset_entry["index"],
                        {
                            "question": question,
                            "superset": superset_entry,
                            "subsets": [],
                        },
                    )

                    subset_indices = {entry["index"] for entry in result["subsets"]}
                    if subset_entry["index"] not in subset_indices:
                        result["subsets"].append(subset_entry)

    return list(results_by_superset.values())


def load_dataset_rows() -> List[Dict]:
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    except Exception as error:
        print("Standard dataset load failed, retrying with streaming mode:", error)
        test_stream = load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=True)
        validation_stream = load_dataset("TIGER-Lab/MMLU-Pro", split="validation", streaming=True)

        test_rows = list(test_stream)
        validation_rows = list(validation_stream)

        print("number of test examples:", len(test_rows))
        print("number of validation examples:", len(validation_rows))
        return test_rows + validation_rows

    print("number of test examples:", dataset["test"].num_rows)
    print("number of validation examples:", dataset["validation"].num_rows)

    combined = concatenate_datasets([dataset["test"], dataset["validation"]])
    return combined.to_list()


if __name__ == "__main__":
    data = load_dataset_rows()
    supersets = find_supersets(data)

    supersets_map = {}
    for item in supersets:
        superset = item["superset"]
        subset_ids = [subset["question_id"] for subset in item["subsets"] if subset["question_id"] is not None]
        question_id = superset.get("question_id")
        if question_id is None or not subset_ids:
            continue

        key = str(question_id)
        existing = supersets_map.setdefault(key, [])
        for subset_id in subset_ids:
            if subset_id not in existing:
                existing.append(subset_id)

    print(json.dumps({"supersets": supersets_map}, ensure_ascii=False, indent=2))
    print(f"Found {len(supersets_map)} superset question rows.")
