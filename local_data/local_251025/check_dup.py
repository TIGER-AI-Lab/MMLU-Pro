from collections import defaultdict
from itertools import combinations
from datasets import load_dataset, concatenate_datasets


def find_pairwise_duplicates(data):
    duplicates = []
    seen_questions = defaultdict(list)

    for i, item in enumerate(data):
        question = item["question"].strip()
        seen_questions[question].append(i)

    # Now check for duplicates
    for question, indices in seen_questions.items():
        if len(indices) > 1:
            # Check all pairs in this group
            for idx1, idx2 in combinations(indices, 2):
                # Check if options match
                options1 = set(val for val in data[idx1]["options"] if val is not None)
                options2 = set(val for val in data[idx2]["options"] if val is not None)

                if options1 == options2:
                    duplicates.append({
                        "question": question,
                        "indices": [idx1, idx2],
                        "question_ids": [data[idx1]["question_id"], data[idx2]["question_id"]]
                    })

    return duplicates


if __name__ == "__main__":
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    combined = concatenate_datasets([dataset["test"], dataset["validation"]])
    data = combined.to_list()

    duplicates = find_pairwise_duplicates(data)

    for dup in duplicates:
        # print(f"\nQuestion: {dup['question']}")
        print(f"Question IDs: {dup['question_ids']}")
    print(f"Found {len(duplicates)} duplicate pairs:")
