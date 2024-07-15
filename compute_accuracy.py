import glob
import sys
import json
import re
import random

assert len(sys.argv) > 1, 'You need to pass the directory'
path = sys.argv[1]


def extract_answer(text, level):
    if level == 'l1':
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return None
    elif level == 'l2':
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)
    

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


for name in glob.glob(path + '/*'):
    print('Level 1 regex' + '==' * 20)
    succ, fail = 0, 0
    with open(name, 'r') as f:
        entries = json.load(f)
        for e in entries:
            pred = extract_answer(e['model_outputs'], 'l1')
            if pred is None:
                random.seed(12345)
                pred = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
            # Remove the None cases
            if pred == e['answer']:
                succ += 1
            else:
                fail += 1
    print(name, succ / (succ + fail))

    print('Level 2 regex' + '==' * 20)
    succ, fail = 0, 0
    with open(name, 'r') as f:
        entries = json.load(f)
        for e in entries:
            pred = extract_answer(e['model_outputs'], 'l2')
            if pred is None:
                random.seed(12345)
                pred = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
            # Remove the None cases
            if pred == e['answer']:
                succ += 1
            else:
                fail += 1
    print(name, succ / (succ + fail))
    
    print()