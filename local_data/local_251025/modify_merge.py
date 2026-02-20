import json


with open("mmlu_pro_test_data_ori.json", "r") as fi:
    data = json.load(fi)

modified = []
modified_map = {}
with open("merge_modify.jsonl", "r") as fi:
    for line in fi.readlines():
        curr = json.loads(line)
        modified.append(curr)
        modified_map[curr["question_id"]] = curr

count = 0
for i, each in enumerate(data):
    if each["question_id"] in modified_map:
        data[i] = modified_map[each["question_id"]]
        count += 1

print(count)

with open("mmlu_pro_test_data_1stage.json.json", "w") as fo:
    fo.write(json.dumps(data))




