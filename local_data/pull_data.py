from datasets import load_dataset
from datasets import Dataset
import json
import os


def deduplicate(data):
    map_result = []
    count = 0
    temp = []
    for item in data:
        key = (item["category"], item["question"], item["answer"], ", ".join(item["options"]))
        if key not in map_result:
            map_result.append(key)
            temp.append(item)
        else:
            count += 1
            print(key)
    return temp


def pull_data():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_data, val_data = [], []
    for each in dataset["test"]:
        test_data.append(each)
    for each in dataset["validation"]:
        val_data.append(each)
    # test_data = deduplicate(test_data)
    # val_data = deduplicate(val_data)
    sta_subjects = {}
    for each in dataset["test"]:
        if each["category"] not in sta_subjects:
            sta_subjects[each["category"]] = 0
        sta_subjects[each["category"]] += 1
    print(sta_subjects)
    # sta_source(test_data)
    os.makedirs("local_251026/", exist_ok=True)
    with open("local_260118/mmlu_pro_test_data_ori.json", "w") as fo:
        fo.write(json.dumps(test_data))
    with open("local_260118/mmlu_pro_val_data.json", "w") as fo:
        fo.write(json.dumps(val_data))
    return test_data, val_data


def update_data(data):
    return data


def push_data(data):
    test_data, val_data = data
    test_dataset = Dataset.from_list(test_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset.push_to_hub("TIGER-Lab/MMLU-Pro", split="test")
    val_dataset.push_to_hub("TIGER-Lab/MMLU-Pro", split="validation")


def sta_source(test_data):
    src_sta_map = {}
    for each in test_data:
        cat = each["category"]
        if cat not in src_sta_map:
            src_sta_map[cat] = {"ori_mmlu": 0, "new": 0}
        if "ori_mmlu" in each["src"]:
            src_sta_map[cat]["ori_mmlu"] += 1
        else:
            src_sta_map[cat]["new"] += 1
    print(src_sta_map)


if __name__ == "__main__":
    hf_data = pull_data()
    # push_data(hf_data)



