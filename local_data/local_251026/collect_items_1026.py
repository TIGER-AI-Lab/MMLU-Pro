import json



target_ids = (6717, 6079, 6205, 6394, 6645, 6018, 6021, 6077, 6091, 6092, 6093, 6094, 6150, 6153, 6194, 6195, 6205, 6214, 6215, 6287, 6288, 6292, 6320, 6328, 6337, 6340, 6341, 6342, 6384, 6404, 6405, 6407, 6410, 6413, 6414, 6459, 6469, 6470, 6585, 6598, 6623, 6624, 6645, 6647, 6656, 6657, 6732, 6738, 6739, 6770)

with open("../local_251024/mmlu_pro_test_data.json", "r") as fi:
    data = json.load(fi)


collected_items = []
for each in data:
    if each["question_id"] in target_ids:
        collected_items.append(each)

print("len(collected_items):", len(collected_items))
with open("collected_items_1026.json", "w") as fo:
    fo.write(json.dumps(collected_items))




