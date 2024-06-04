# MMLU-Pro

|[**🤗 Dataset**](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | [**🏆Leaderboard**](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro) | [**📖 Paper**](https://arxiv.org/abs/2406.01574) |

This repo contains the evaluation code for the paper "[MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](https://arxiv.org/abs/2406.01574.pdf)"

## Introduction
We introduce MMLU-Pro, an enhanced benchmark designed to evaluate language understanding models across broader and more challenging tasks. Building on the Massive Multitask Language Understanding (MMLU) dataset, MMLU-Pro integrates more challenging, reasoning-focused questions and increases the answer choices per question from four to ten, significantly raising the difficulty and reducing the chance of success through random guessing. MMLU-Pro comprises over 12,000 rigorously curated questions from academic exams and textbooks, spanning 14 diverse domains including Biology, Business, Chemistry, Computer Science, Economics, Engineering, Health, History, Law, Math, Philosophy, Physics, Psychology, and Others.  

Our experimental results show that MMLU-Pro not only raises the challenge, causing a significant drop in accuracy by 16% to 33% compared to MMLU but also demonstrates greater stability under varying prompts. With 24 different prompt styles tested, the sensitivity of model scores to prompt variations decreased from 4-5% in MMLU to just 2% in MMLU-Pro. Additionally, we found that models utilizing Chain of Thought (CoT) reasoning achieved better performance on MMLU-Pro compared to direct answering, which starkly contrasts the findings on the original MMLU, indicating that MMLU-Pro includes more complex reasoning questions. 

<img width="1432" alt="abs" src="https://github.com/TIGER-AI-Lab/MMLU-Pro/assets/20929360/8e369fc2-5b6b-4bab-8a44-9e222e742027">

## Dataset Creation
MMLU-Pro was created to provide language models with a more challenging and robust benchmark, pushing the boundaries of what these models can achieve in terms of expert-level knowledge and reasoning. Please refer to our huggingface [**🤗 Dataset**](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) for more details.

## Evaluation

To run local inference, modify the model name in the following script and execute it:

```bash
cd scripts/examples/
sh eval_llama_2_7b.sh
```

To use the API for inference, modify the API KEY and client creation in the following script and execute it:

```bash
cd scripts/examples/
sh eval_gpt_4.sh
```
## 🏆 Mini-Leaderboard
| Model                          | Overall Accuracy | 
|--------------------------------|:----------------:|
| GPT-4o                         | 72.55            | 
| Gemini-1.5-Pro                 | 69.03            |
| Claude-3-Opus                  | 68.45            |
| GPT-4-Turbo                    | 63.71            | 
| Gemini-1.5-Flash               | 59.12            |
| Yi-large                       | 57.53            |
| Claude-3-Sonnet                | 56.80            |
| Llama-3-70B-Instruct           | 56.20            |
| Phi3-medium-4k                 | 55.70            |
| Deepseek-V2-Chat               | 54.81            |
| Phi-3-medium-4k-instruct       | 53.48            |
| Llama-3-70B                    | 52.78            |
| Qwen1.5-72B-Chat               | 52.64            |
| Yi-1.5-34B-Chat                | 52.29            |
| Phi3-medium-128k               | 51.91            |
| MAmmoTH2-8x7B-Plus             | 50.40            |

For more details on various models and their accuracy across different subjects, please visit our [**Leaderboard**](https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro).

## Contact
- Yubo Wang: y726wang@uwaterloo.ca
- Xueguang Ma: x93ma@uwaterloo.ca
- Wenhu Chen: wenhuchen@uwaterloo.ca

## Citation

**BibTeX:**
```bibtex
@misc{wang2024mmlupro,
      title={MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark}, 
      author={Yubo Wang and Xueguang Ma and Ge Zhang and Yuansheng Ni and Abhranil Chandra and Shiguang Guo and Weiming Ren and Aaran Arulraj and Xuan He and Ziyan Jiang and Tianle Li and Max Ku and Kai Wang and Alex Zhuang and Rongqi Fan and Xiang Yue and Wenhu Chen},
      year={2024},
      eprint={2406.01574},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
