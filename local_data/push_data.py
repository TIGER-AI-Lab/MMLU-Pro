from datasets import load_dataset
from datasets import Dataset
import json


def push_data():
    # with open("pushed_data/test_data.json", "r") as fi:
    with open("local_260118/mmlu_pro_test_data_new.json", "r") as fi:
        test_data = json.load(fi)
    with open("local_260118/mmlu_pro_val_data.json", "r") as fi:
        val_data = json.load(fi)
    for i, each in enumerate(test_data):
        if "cot_content" not in each:
            test_data[i]["cot_content"] = ""

    sta_subjects = {}
    for each in test_data:
        if each["category"] not in sta_subjects:
            sta_subjects[each["category"]] = 0
        sta_subjects[each["category"]] += 1
    print(sta_subjects)
    print("length of test_data", len(test_data))
    test_dataset = Dataset.from_list(test_data)
    # val_dataset = Dataset.from_list(val_data)
    test_dataset.push_to_hub("TIGER-Lab/MMLU-Pro", split="test")
    # val_dataset.push_to_hub("TIGER-Lab/MMLU-Pro", split="validation")


if __name__ == "__main__":
    push_data()



