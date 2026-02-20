import json


with open("mmlu_pro_test_data_1stage.json", "r") as fi:
    data = json.load(fi)

# modified = []
modified_map = {}
with open("modified_again.json", "r") as fi:
    modified = json.load(fi)
    for each in modified:
        modified_map[each["question_id"]] = each

count = 0
for i, each in enumerate(data):
    if each["question_id"] in modified_map:
        data[i] = modified_map[each["question_id"]]
        count += 1

print(count)

with open("mmlu_pro_test_data_2stage.json", "w") as fo:
    fo.write(json.dumps(data))




