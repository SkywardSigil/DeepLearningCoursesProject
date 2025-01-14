# preprocess_GPT2.py
import os
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import argparse

def load_humaneval(data_dir):
    file_path = os.path.join(data_dir, "data", "HumanEval.jsonl")
    examples = []
    if not os.path.exists(file_path):
        print(f"错误：HumanEval 数据文件 {file_path} 不存在。")
        return DatasetDict({"train": Dataset.from_list(examples)})
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            input_text = obj.get("prompt", "")
            output_text = obj.get("completion", "")
            test_cases = obj.get("test", [])  # 提取测试用例
            examples.append({
                "input": input_text, 
                "output": output_text,
                "test_cases": test_cases  # 包含测试用例
            })
    return DatasetDict({
        "train": Dataset.from_list(examples)  # HumanEval 通常只有一个训练集
    })

def load_apps(data_dir):
    splits = ["train", "test"]
    data = {}
    for split in splits:
        file_path = os.path.join(data_dir, f"{split}.jsonl")
        examples = []
        if not os.path.exists(file_path):
            print(f"警告：APPS 数据文件 {file_path} 不存在，跳过。")
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                input_text = obj.get("prompt", "")
                output_text = obj.get("solution", "")
                examples.append({"input": input_text, "output": output_text})
        data[split] = Dataset.from_list(examples)
    return DatasetDict({
        "train": data.get("train", Dataset.from_list([])),
        "validation": data.get("test", Dataset.from_list([]))  # APPS没有单独的validation集，使用test集作为验证
    })

def load_codexglue_codecompletion_token(data_dir):
    """
    加载 CodeXGLUE CodeCompletion-token 任务的数据集。
    假设每个子目录（如 javaCorpus 和 py150）中的 literals.json 文件包含代码片段字符串。
    将每个代码片段拆分为 input 和 output 两部分。
    """
    subdirs = ["javaCorpus", "py150"]
    all_examples = []
    for subdir in subdirs:
        dataset_dir = os.path.join(data_dir, subdir)
        literals_file = os.path.join(dataset_dir, "literals.json")
        if not os.path.exists(literals_file):
            print(f"警告：文件 {literals_file} 不存在，跳过。")
            continue
        with open(literals_file, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
                print(f"加载 {literals_file}，包含 {len(json_data)} 个条目。")
                for item in json_data:
                    if isinstance(item, str):
                        # 将代码片段拆分为 input 和 output
                        split_ratio = 0.8  # 80% 作为 input，20% 作为 output
                        split_idx = int(len(item) * split_ratio)
                        input_text = item[:split_idx]
                        output_text = item[split_idx:]
                        all_examples.append({"input": input_text, "output": output_text})
                    else:
                        print(f"警告：条目类型为 {type(item)}，预期为 str，跳过。")
            except json.JSONDecodeError as e:
                print(f"错误：无法解析文件 {literals_file}: {e}")
    print(f"CodeXGLUE CodeCompletion-token 加载了 {len(all_examples)} 个例子。")
    return DatasetDict({
        "train": Dataset.from_list(all_examples)  # CodeCompletion-token 通常只有训练集
    })

def tokenize_function(examples, tokenizer, max_length=512):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(
        inputs, 
        max_length=max_length, 
        truncation=True, 
        padding='max_length'
    )
    # GPT-2 不支持 `as_target_tokenizer()`，所以直接编码标签
    labels = tokenizer(
        targets, 
        max_length=max_length, 
        truncation=True, 
        padding='max_length'
    )
    model_inputs["labels"] = labels["input_ids"]
    # 保留 test_cases 不进行分词
    if "test_cases" in examples:
        model_inputs["test_cases"] = examples["test_cases"]
    return model_inputs

def preprocess_and_tokenize(dataset_dict, tokenizer, output_dir, max_length=512):
    # 检查 dataset_dict 是否为空
    if len(dataset_dict["train"]) == 0:
        print(f"警告：数据集 {output_dir} 为空，跳过预处理。")
        return
    tokenized_datasets = dataset_dict.map(
        lambda x: tokenize_function(x, tokenizer, max_length), 
        batched=True, 
        remove_columns=["input", "output"]
    )
    tokenized_datasets.save_to_disk(output_dir)
    print(f"已保存预处理数据到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="预处理数据集")
    parser.add_argument("--humaneval_dir", type=str, required=True, help="HumanEval数据集目录")
    parser.add_argument("--apps_dir", type=str, required=True, help="APPS数据集目录")
    parser.add_argument("--codexglue_dir", type=str, required=True, help="CodeXGLUE CodeCompletion-token任务数据集目录")
    parser.add_argument("--output_preprocessed_path", type=str, default="results/preprocessed", help="预处理后数据集保存路径")
    
    # 将 model_name 默认值更改为本地分词器路径
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=r"models/gpt2_local", 
        help="预训练模型名称或本地路径"
    )
    
    
    args = parser.parse_args()
    
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # GPT-2 默认不使用 `eos_token` 的填充，因此需要设置 `padding_side` 和其他相关参数
    # 这里我们确保 GPT-2 使用 `eos_token` 作为填充 token
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载并预处理HumanEval
    print("加载HumanEval数据集...")
    humaneval = load_humaneval(args.humaneval_dir)
    print("预处理HumanEval数据集...")
    preprocess_and_tokenize(
        humaneval, 
        tokenizer, 
        os.path.join(args.output_preprocessed_path, "HumanEval")
    )
    
    # 加载并预处理APPS
    print("加载APPS数据集...")
    apps = load_apps(args.apps_dir)
    print("预处理APPS数据集...")
    preprocess_and_tokenize(
        apps, 
        tokenizer, 
        os.path.join(args.output_preprocessed_path, "APPS")
    )
    
    # 加载并预处理CodeXGLUE CodeCompletion-token
    print("加载CodeXGLUE CodeCompletion-token数据集...")
    codexglue = load_codexglue_codecompletion_token(args.codexglue_dir)
    if len(codexglue["train"]) == 0:
        print("错误：CodeXGLUE CodeCompletion-token 数据集为空，请检查数据格式和路径。")
        return
    print("预处理CodeXGLUE CodeCompletion-token数据集...")
    preprocess_and_tokenize(
        codexglue, 
        tokenizer, 
        os.path.join(args.output_preprocessed_path, "CodeXGLUE_CodeCompletion_token")
    )
    
    print("所有数据集预处理完成。")

if __name__ == "__main__":
    main()
