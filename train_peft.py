# train_peft.py

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import (
    get_peft_model,
    LoraConfig,
    PromptTuningConfig,
    PrefixTuningConfig,
    IA3Config,
    TaskType
)
from torch.utils.data import DataLoader
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
import csv
import os
import torch.cuda.amp as amp

# 将 collate_fn 移动到顶层并修改为使用 torch.tensor
def collate_fn(batch):
    return {key: torch.tensor([item[key] for item in batch]) for key in batch[0]}

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
    parser.add_argument("--prompt_length", type=int, default=10, help="Prompt length for Prompt Tuning")
    # Prefix Tuning specific
    parser.add_argument("--prefix_length", type=int, default=10, help="Prefix length for Prefix Tuning")
    # IA3 specific
    parser.add_argument("--ia3_hidden_size", type=int, default=64, help="IA3 hidden size")
    # IA3 specific additional parameters can be added here if needed
    parser.add_argument("--output_model_path", type=str, required=True, help="Path to save the trained model")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory for TensorBoard logs")
    parser.add_argument("--preprocessed_data_path", type=str, default="results/preprocessed", help="Path to preprocessed data")
    parser.add_argument("--csv_path", type=str, default="results/experiment_results.csv", help="Path to results CSV file")
    parser.add_argument("--method_name", type=str, required=True, help="Name of the PEFT method")
    return parser.parse_args()

def load_datasets(preprocessed_data_path):
    print("Loading preprocessed datasets...")
    # 根据预处理后的目录结构加载数据集
    humaneval_path = os.path.join(preprocessed_data_path, "HumanEval")
    apps_path = os.path.join(preprocessed_data_path, "APPS")
    codexglue_path = os.path.join(preprocessed_data_path, "CodeXGLUE_CodeCompletion_token")
    
    # 加载各个数据集
    humaneval = load_from_disk(humaneval_path)
    apps = load_from_disk(apps_path)
    codexglue = load_from_disk(codexglue_path)
    
    # 获取训练集
    train_humaneval = humaneval["train"]
    train_apps = apps["train"]
    train_codexglue = codexglue["train"]
    
    # 使用 concatenate_datasets 合并训练集
    train_combined = concatenate_datasets([train_humaneval, train_apps, train_codexglue])
    
    # 获取验证集，以 APPS 的验证集为主
    validation = apps["validation"]  # APPS的validation集
    
    combined_dataset = DatasetDict({
        "train": train_combined,
        "validation": validation
    })
    
    return combined_dataset

def get_peft_model_instance(method, model, args, tokenizer):
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
        prompt_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_length=args.prompt_length,
            tokenizer=tokenizer,
        )
        model = get_peft_model(model, prompt_config)
    elif method == "PrefixTuning":
        prefix_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_prefix_tokens=args.prefix_length
        )
        model = get_peft_model(model, prefix_config)
    elif method == "IA3":
        ia3_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            hidden_size=args.ia3_hidden_size
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
    # 使用 text_target 参数进行标签的编码
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=max_length, 
            truncation=True, 
            padding='max_length'
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    
    # 加载数据集
    dataset = load_datasets(args.preprocessed_data_path)
    
    # 加载分词器和模型
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("E:/NJU/深度学习平台/final/exp_new/models/codebert-base")
    model = AutoModelForCausalLM.from_pretrained(
        "E:/NJU/深度学习平台/final/exp_new/models/codebert-base",
        # 对于生成任务，通常不需要指定 num_labels
    )
    
    # 获取PEFT模型实例
    model = get_peft_model_instance(args.method, model, args, tokenizer)
    
    # 准备数据加载器
    print("Preparing data loaders...")
    dataloader = {
        "train": DataLoader(
            dataset["train"], 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn, 
            num_workers=4
        ),
        "validation": DataLoader(
            dataset["validation"], 
            batch_size=args.batch_size, 
            collate_fn=collate_fn, 
            num_workers=4
        )
    }
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 优化器（仅优化需要的参数）
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 混合精度训练的GradScaler
    scaler = amp.GradScaler()
    
    # TensorBoard记录
    writer = SummaryWriter(args.log_dir)
    
    # 检查并创建CSV文件
    if not os.path.exists(args.csv_path):
        with open(args.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow(['method', 'epoch', 'split', 'loss', 'perplexity'])  # 修改为perplexity
    
    # 训练参数
    num_epochs = args.num_epochs
    accumulation_steps = args.accumulation_steps
    patience = 2
    best_metric = float('inf')  # 对于生成任务，通常使用损失或困惑度
    patience_counter = 0
    
    # 训练函数
    def train_model(dataloader, model, optimizer, writer, num_epochs=3):
        nonlocal best_metric, patience_counter
        model.train()
        global_step = 0
        for epoch in range(num_epochs):
            loop = tqdm(dataloader["train"], desc=f"Epoch {epoch+1}/{num_epochs}")
            optimizer.zero_grad()
            for i, batch in enumerate(loop):
                try:
                    # 将batch转移到设备
                    batch = {key: value.to(device) for key, value in batch.items()}
                    
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
                    print(f"Training error: {e}")
                    return  # 退出训练
            
            # 验证阶段
            validate_model(dataloader, model, writer, epoch, num_epochs, args.method_name)
            
            # 早停检查
            current_metric = validate_model.current_metric
            if current_metric < best_metric:
                best_metric = current_metric
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    
    # 验证函数
    def validate_model(dataloader, model, writer, epoch, num_epochs, method_name="PEFT"):
        model.eval()
        total_loss = 0
        total_examples = 0
        with torch.no_grad():
            for batch in dataloader["validation"]:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item() * batch["input"].size(0)  # 使用 'input' 代替 'input_ids'
                total_examples += batch["input"].size(0)
        
        avg_loss = total_loss / total_examples
        perplexity = torch.exp(torch.tensor(avg_loss))
        validate_model.current_metric = avg_loss  # 使用平均损失作为度量
        
        writer.add_scalar("Validation/Loss", avg_loss, epoch * num_epochs)
        writer.add_scalar("Validation/Perplexity", perplexity.item(), epoch * num_epochs)
        print(f"Validation - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        # 写入CSV
        with open(args.csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([method_name, epoch + 1, "validation", avg_loss, perplexity.item()])
        
        model.train()
        return avg_loss
    
    # 开始训练
    print("Starting training...")
    try:
        train_model(dataloader, model, optimizer, writer, num_epochs=num_epochs)
    except Exception as e:
        print(f"Training error: {e}")
    
    # 保存模型
    print(f"Saving model to {args.output_model_path}...")
    model.save_pretrained(args.output_model_path)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
