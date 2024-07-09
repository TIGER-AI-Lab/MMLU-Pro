import json
import os


def merge_result(dir_path, output_path):
    res = []
    for file in os.listdir(dir_path):
        if not file.endswith("result.json"):
            continue
        file_path = os.path.join(dir_path, file)
        with open(file_path, "r") as fi:
            curr = json.load(fi)
        res += curr
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res, indent=2))


merge_result("../eval_results/gpt4o/", "../eval_results/model_outputs_gpt4o_5shots.json")

merge_result("../eval_results/deepseek/", "../eval_results/model_outputs_deepseek_5shots.json")
