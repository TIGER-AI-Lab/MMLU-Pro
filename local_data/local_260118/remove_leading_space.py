import json
import argparse
from pathlib import Path


def analyze_leading_spaces(data):
    """分析数据中有leading space的选项"""
    issues = []

    for item in data:
        question_id = item.get("question_id")
        options = item.get("options", [])
        category = item.get("category", "unknown")

        for idx, option in enumerate(options):
            if isinstance(option, str) and option != option.lstrip():
                leading_spaces = len(option) - len(option.lstrip())
                issues.append({
                    "question_id": question_id,
                    "category": category,
                    "option_index": idx,
                    "option_letter": chr(65 + idx),  # A, B, C, ...
                    "leading_spaces": leading_spaces,
                    "original": repr(option),
                    "fixed": repr(option.lstrip())
                })

    return issues


def fix_leading_spaces(data):
    """修复所有选项中的leading space"""
    fixed_count = 0

    for item in data:
        options = item.get("options", [])
        for idx, option in enumerate(options):
            if isinstance(option, str) and option != option.lstrip():
                options[idx] = option.lstrip()
                fixed_count += 1

    return data, fixed_count


def main():
    parser = argparse.ArgumentParser(description="检测并修复MMLU-Pro数据中选项的leading space问题")
    parser.add_argument("-i", "--input", default="mmlu_pro_test_data_ori.json", help="输入的JSON文件路径")
    parser.add_argument("-o", "--output", default="mmlu_pro_test_data_new.json", help="输出的JSON文件路径")
    parser.add_argument("--analyze-only", action="store_true", help="仅分析，不修复")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")

    args = parser.parse_args()
    input_path = Path(args.input)  # 注意这里改成 args.input

    print(f"读取文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"共加载 {len(data)} 条数据")

    # 分析问题
    issues = analyze_leading_spaces(data)

    if not issues:
        print("✅ 未发现leading space问题！")
        return

    # 统计
    affected_questions = len(set(i["question_id"] for i in issues))
    by_category = {}
    for issue in issues:
        cat = issue["category"]
        by_category[cat] = by_category.get(cat, 0) + 1

    print(f"\n⚠️  发现 {len(issues)} 个选项存在leading space问题")
    print(f"   涉及 {affected_questions} 个问题")
    print(f"\n按类别统计:")
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        print(f"   {cat}: {count} 个选项")

    if args.verbose:
        print(f"\n详细列表:")
        for issue in issues[:50]:  # 只显示前50个
            print(f"   Q{issue['question_id']} [{issue['category']}] "
                  f"选项{issue['option_letter']}: {issue['original']} -> {issue['fixed']}")
        if len(issues) > 50:
            print(f"   ... 还有 {len(issues) - 50} 个")

    if args.analyze_only:
        print("\n(仅分析模式，未进行修复)")
        return

    # 修复
    fixed_data, fixed_count = fix_leading_spaces(data)

    # 保存
    output_path = Path(args.output) if args.output else input_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已修复 {fixed_count} 个选项")
    print(f"   保存到: {output_path}")


if __name__ == "__main__":
    main()