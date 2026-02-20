import json



target_question_pairs = [
    (1137, 1138),
    (1150, 1149),
    (1695, 1696),
    (1851, 1852),
    (1868, 1869),
    (6194, 6195),
    (6321, 6322),
    (6371, 6372),
    (6373, 6374),
    (6685, 6686),
    (6704, 6703)
]

target_ids = []
target = target_question_pairs
for each in target:
    target_ids.append(each[1])

with open("../local_251024/mmlu_pro_test_data.json", "r") as fi:
    data = json.load(fi)


collected_items = []
for each in data:
    if each["question_id"] in target_ids:
        collected_items.append(each)

print("len(collected_items):", len(collected_items))
with open("collected_items_again.json", "w") as fo:
    fo.write(json.dumps(collected_items))




