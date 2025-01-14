import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, DataCollatorWithPadding
from peft import (
    get_peft_model,
    LoraConfig,
    PromptTuningConfig,
    PrefixTuningConfig,
    IA3Config,
    TaskType
)
from torch.utils.data import DataLoader
from datasets import load_from_disk, DatasetDict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
import csv
import os
import torch.cuda.amp as amp
import subprocess
import sys
import tempfile
import io
import re
import logging

# 重新配置标准输出和标准错误输出为 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # 使用重新配置的 stdout
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train PEFT methods on code generation tasks")
    parser.add_argument("--method", type=str, required=True, choices=["LoRA", "PromptTuning", "PrefixTuning", "IA3"], help="PEFT method to use")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU/CPU")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps")
    # LoRA specific
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    # Prompt Tuning specific
    parser.add_argument("--num_virtual_tokens", type=int, default=10, help="Number of virtual tokens for Prompt Tuning")
    # Prefix Tuning specific
    parser.add_argument("--num_prefix_tokens", type=int, default=10, help="Number of prefix tokens for Prefix Tuning")
    # IA3 specific
    parser.add_argument("--ia3_target_modules", type=str, nargs='+', default=["attn.c_proj", "mlp.c_fc", "mlp.c_proj"], help="Modules to apply IA3 to")
    parser.add_argument("--ia3_feedforward_modules", type=str, nargs='+', default=["mlp.c_fc", "mlp.c_proj"], help="Feedforward modules to apply IA3 to")
    # Added Pass@K parameter
    parser.add_argument("--pass_k", type=int, default=5, help="K value for Pass@K metric")
    # Removed BitFit specific arguments

    parser.add_argument("--output_model_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory for TensorBoard logs")
    parser.add_argument("--preprocessed_data_path", type=str, required=True, help="Path to preprocessed data")
    parser.add_argument("--csv_path", type=str, default="results/experiment_results.csv", help="Path to results CSV file")
    parser.add_argument("--method_name", type=str, required=True, help="Name of the PEFT method")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the pre-trained model")
    return parser.parse_args()

def load_datasets(preprocessed_data_path):
    logging.info(f"Trying to load dataset from: {preprocessed_data_path}")
    if not os.path.exists(preprocessed_data_path):
        raise FileNotFoundError(f"Preprocessed data path {preprocessed_data_path} does not exist.")
    dataset = load_from_disk(preprocessed_data_path)
    # Check if 'validation' split exists
    if 'validation' not in dataset:
        logging.info(f"数据集 {preprocessed_data_path} 没有 validation 划分，使用 train 作为 validation")
        dataset = DatasetDict({
            "train": dataset["train"],
            "validation": dataset["train"]
        })
    logging.info(f"Loaded dataset splits: {dataset.keys()}")
    return dataset

def get_peft_model_instance(method, model, args, tokenizer):
    logging.info(f"Initializing PEFT method: {method}")
    if method == "LoRA":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        model = get_peft_model(model, peft_config)
    elif method == "PromptTuning":
        logging.info(f"Configuring PromptTuning with num_virtual_tokens={args.num_virtual_tokens}")
        prompt_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.num_virtual_tokens
            # Removed tokenizer argument
        )
        model = get_peft_model(model, prompt_config)
    elif method == "PrefixTuning":
        logging.info(f"Configuring PrefixTuning with num_prefix_tokens={args.num_prefix_tokens}")
        # 将 num_prefix_tokens 映射到 num_virtual_tokens
        prefix_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.num_prefix_tokens  # 使用 num_virtual_tokens 替代 num_prefix_tokens
        )
        model = get_peft_model(model, prefix_config)
    elif method == "IA3":
        logging.info(f"Configuring IA3 with target_modules={args.ia3_target_modules} and feedforward_modules={args.ia3_feedforward_modules}")
        ia3_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=args.ia3_target_modules,
            feedforward_modules=args.ia3_feedforward_modules,
            init_ia3_weights=True
        )
        model = get_peft_model(model, ia3_config)
    else:
        raise ValueError("Unsupported method")
    return model

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

def run_tests(code, test_cases):
    """
    运行生成的代码并测试其是否通过给定的测试用例。

    参数：
    - code (str): 生成的代码片段。
    - test_cases (list of str): 测试用例列表。

    返回：
    - bool: 如果代码通过所有测试用例，则返回 True，否则返回 False。
    """
    logging.info("开始运行测试用例...")

    # 清理生成的代码，移除 '>>>' 提示符和多余的注释
    cleaned_code = "\n".join(
        line for line in code.splitlines() 
        if not line.strip().startswith(">>>") 
        and not line.strip().startswith("**") 
        and not line.strip().startswith("*")
    )

    # 使用正则表达式移除行尾的注释
    cleaned_code = re.sub(r'#.*', '', cleaned_code)

    # 移除多余的空行
    cleaned_code = "\n".join([line for line in cleaned_code.splitlines() if line.strip() != ""])

    # 将代码和测试用例写入临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(cleaned_code)
        tmp_file_path = tmp_file.name
    logging.info(f"代码已写入临时文件: {tmp_file_path}")

    try:
        # 使用 tqdm 显示测试用例的进度
        for idx, test in enumerate(tqdm(test_cases, desc="运行测试用例", unit="用例")):
            # logging.info(f"运行测试用例 {idx + 1}/{len(test_cases)}: {test}")
            # 构建测试命令
            try:
                result = subprocess.run(
                    [sys.executable, tmp_file_path, test],
                    capture_output=True,
                    text=True,
                    timeout=10,  # 设置超时时间（秒）
                    encoding='utf-8'
                )
                if result.returncode != 0:
                    # logging.warning(f"测试用例 {idx + 1} 失败: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                # logging.error(f"测试用例 {idx + 1} 超时.")
                return False
            except Exception as e:
                # logging.error(f"运行测试用例 {idx + 1} 时发生错误: {e}")
                return False
    finally:
        os.remove(tmp_file_path)
        logging.info(f"已删除临时文件: {tmp_file_path}")

    # logging.info("所有测试用例通过.")
    return True

def validate_model(dataloader, model, tokenizer, device, pass_k, writer, epoch, num_epochs, method_name, csv_path):
    model.eval()
    total_loss = 0
    total_examples = 0
    total_pass = 0
    with torch.no_grad():
        for batch in dataloader["validation"]:
            # 将batch转移到设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            test_cases_list = batch.get("test_cases", None)
            
            if test_cases_list is None:
                logging.error("Error: 'test_cases' field not found in the dataset.")
                continue  # 跳过该批次
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_examples += input_ids.size(0)
            
            # 获取输入文本和对应的测试用例
            inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            test_cases = test_cases_list  # 已经是列表
            
            # 添加 tqdm 进度条到外层循环
            for input_text, test_case in tqdm(zip(inputs, test_cases), total=len(inputs), desc="Validating examples"):
                # 生成 K 个代码片段
                generated_codes = []
                for _ in range(pass_k):
                    input_ids_gen = tokenizer.encode(input_text, return_tensors='pt').to(device)
                    try:
                        generated_ids = model.generate(
                            input_ids_gen,
                            max_length=512,
                            num_return_sequences=1,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                        generated_codes.append(generated_code)
                    except Exception as e:
                        logging.error(f"生成代码时发生错误: {e}")
                        continue  # 跳过当前生成

                # 评估生成的代码
                pass_k_flag = False
                for code in generated_codes:
                    if run_tests(code, test_case):
                        pass_k_flag = True
                        break
                if pass_k_flag:
                    total_pass += 1

    avg_loss = total_loss / total_examples if total_examples > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)) if avg_loss > 0 else float('inf')
    pass_at_k = total_pass / total_examples if total_examples > 0 else 0
    logging.info(f"Validation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}, Pass@{pass_k}: {pass_at_k:.4f}")
    
    # 记录到 TensorBoard
    writer.add_scalar("Validation/Loss", avg_loss, epoch * num_epochs)
    writer.add_scalar("Validation/Perplexity", perplexity.item(), epoch * num_epochs)
    writer.add_scalar(f"Validation/Pass@{pass_k}", pass_at_k, epoch * num_epochs)
    
    # 写入CSV
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow([method_name, epoch + 1, "validation", avg_loss, perplexity.item(), pass_at_k])
    
    model.train()
    return avg_loss

def train_model(dataloader, model, optimizer, writer, num_epochs, accumulation_steps, device, pass_k, method_name, csv_path, tokenizer):
    best_metric = float('inf')  # 对于生成任务，通常使用损失或困惑度
    patience = 2
    patience_counter = 0
    global_step = 0

    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        loop = tqdm(dataloader["train"], desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()
        for i, batch in enumerate(loop):
            try:
                # 将batch转移到设备，排除 'test_cases'
                batch = {k: v.to(device) for k, v in batch.items() if k != "test_cases"}
                
                # 前向传播与混合精度
                with amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss / accumulation_steps
                
                # 反向传播
                scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1
                    writer.add_scalar("Train/Loss", loss.item() * accumulation_steps, global_step)
                
                # 更新进度条
                loop.set_postfix(loss=(loss.item() * accumulation_steps))
            
            except Exception as e:
                logging.error(f"Training error: {e}")
                return  # 退出训练
        
        # 验证阶段
        current_metric = validate_model(
            dataloader=dataloader,
            model=model,
            tokenizer=tokenizer,
            device=device,
            pass_k=pass_k,
            writer=writer,
            epoch=epoch,
            num_epochs=num_epochs,
            method_name=method_name,
            csv_path=csv_path
        )
        
        # 早停检查
        if current_metric < best_metric:
            best_metric = current_metric
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logging.info("Early stopping triggered.")
            break

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    
    # 加载数据集
    dataset = load_datasets(args.preprocessed_data_path)
    
    # 加载分词器和模型
    logging.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # 从本地路径加载分词器
    model = GPT2LMHeadModel.from_pretrained(
        args.model_name,  # 从本地路径加载模型
        pad_token_id=tokenizer.eos_token_id  # 设置pad_token_id
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
        logging.info("pad_token was set to eos_token.")
    
    logging.info(f"Pad token: {tokenizer.pad_token}")
    logging.info(f"Pad token ID: {tokenizer.pad_token_id}")
    
    # 获取PEFT模型实例
    model = get_peft_model_instance(args.method, model, args, tokenizer)
    
    # 初始化DataCollator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 定义自定义的 collate_fn
    def collate_fn(batch):
        test_cases = [item['test_cases'] for item in batch]
        # 移除 'test_cases' 字段
        batch = [{k: v for k, v in item.items() if k != 'test_cases'} for item in batch]
        # 使用 DataCollatorWithPadding 对剩余字段进行处理
        collated = data_collator(batch)
        # 将 'test_cases' 添加回批次中
        collated['test_cases'] = test_cases
        return collated
    
    # 准备数据加载器，使用自定义的 collate_fn
    logging.info("Preparing data loaders...")
    dataloader = {
        "train": DataLoader(
            dataset["train"], 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn, 
            num_workers=0  # 设置为0以避免多进程问题
        ),
        "validation": DataLoader(
            dataset["validation"], 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=0  # 设置为0以避免多进程问题
        )
    }
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 优化器（仅优化需要的参数）
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # TensorBoard记录
    writer = SummaryWriter(args.log_dir)
    
    # 检查并创建CSV文件
    if not os.path.exists(args.csv_path):
        with open(args.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow(['method', 'epoch', 'split', 'loss', 'perplexity', 'pass_k'])  # 添加pass_k
    
    # 开始训练
    logging.info("Starting training...")
    try:
        train_model(
            dataloader=dataloader,
            model=model,
            optimizer=optimizer,
            writer=writer,
            num_epochs=args.num_epochs,
            accumulation_steps=args.accumulation_steps,
            device=device,
            pass_k=args.pass_k,
            method_name=args.method_name,
            csv_path=args.csv_path,
            tokenizer=tokenizer
        )
    except Exception as e:
        logging.error(f"Training error: {e}")
    
    # 保存模型
    logging.info(f"Saving model to {args.output_model_path}...")
    model.save_pretrained(args.output_model_path)
    
    logging.info("Training complete!")

if __name__ == "__main__":
    main()
