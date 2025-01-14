@echo off
setlocal enabledelayedexpansion

:: 定义数据集列表（不含 BitFit）
set DATASETS=HumanEval APPS CodeXGLUE_CodeCompletion_token

:: 定义PEFT方法列表（不包括 BitFit）
set METHODS=LoRA PromptTuning PrefixTuning IA3

:: 定义模型路径
set MODEL_PATH=models/codebert-base

:: 定义预处理数据路径
set PREPROCESSED_PATH=results/preprocessed

:: 定义CSV结果文件路径
set CSV_PATH=results/experiment_results.csv

:: 确保CSV文件有表头
if not exist "%CSV_PATH%" (
    echo method,epoch,split,loss,perplexity > "%CSV_PATH%"
)

:: 遍历每个PEFT方法
for %%M in (%METHODS%) do (
    echo ===============================
    echo 正在运行方法 %%M ...
    echo ===============================
    
    :: 遍历每个数据集
    for %%D in (%DATASETS%) do (
        echo 训练方法 %%M 在数据集 %%D 上...
        
        :: 定义输出模型路径和日志目录
        set OUTPUT_MODEL_PATH=results/%%M_finetuned_models/%%D
        set LOG_DIR=results/%%M_logs/%%D
        
        :: 创建输出模型路径和日志目录（如果不存在）
        mkdir "!OUTPUT_MODEL_PATH!" 2>nul
        mkdir "!LOG_DIR!" 2>nul
        
        :: 根据方法设置特定参数并运行训练
        if "%%M"=="LoRA" (
            python train_peft.py ^
                --method LoRA ^
                --num_epochs 3 ^
                --batch_size 16 ^
                --accumulation_steps 2 ^
                --r 4 ^
                --lora_alpha 32 ^
                --lora_dropout 0.1 ^
                --output_model_path "!OUTPUT_MODEL_PATH!" ^
                --log_dir "!LOG_DIR!" ^
                --preprocessed_data_path "%PREPROCESSED_PATH%/%%D" ^
                --csv_path "%CSV_PATH%" ^
                --method_name LoRA ^
                --model_name "%MODEL_PATH%"
        ) else if "%%M"=="PromptTuning" (
            python train_peft.py ^
                --method PromptTuning ^
                --num_epochs 3 ^
                --batch_size 16 ^
                --accumulation_steps 2 ^
                --prompt_length 10 ^
                --output_model_path "!OUTPUT_MODEL_PATH!" ^
                --log_dir "!LOG_DIR!" ^
                --preprocessed_data_path "%PREPROCESSED_PATH%/%%D" ^
                --csv_path "%CSV_PATH%" ^
                --method_name PromptTuning ^
                --model_name "%MODEL_PATH%"
        ) else if "%%M"=="PrefixTuning" (
            python train_peft.py ^
                --method PrefixTuning ^
                --num_epochs 3 ^
                --batch_size 16 ^
                --accumulation_steps 2 ^
                --prefix_length 10 ^
                --output_model_path "!OUTPUT_MODEL_PATH!" ^
                --log_dir "!LOG_DIR!" ^
                --preprocessed_data_path "%PREPROCESSED_PATH%/%%D" ^
                --csv_path "%CSV_PATH%" ^
                --method_name PrefixTuning ^
                --model_name "%MODEL_PATH%"
        ) else if "%%M"=="IA3" (
            python train_peft.py ^
                --method IA3 ^
                --num_epochs 3 ^
                --batch_size 16 ^
                --accumulation_steps 2 ^
                --ia3_hidden_size 64 ^
                --output_model_path "!OUTPUT_MODEL_PATH!" ^
                --log_dir "!LOG_DIR!" ^
                --preprocessed_data_path "%PREPROCESSED_PATH%/%%D" ^
                --csv_path "%CSV_PATH%" ^
                --method_name IA3 ^
                --model_name "%MODEL_PATH%"
        )
        
        echo 方法 %%M 在数据集 %%D 上的训练完成。
        echo.
    )
    
    echo ===============================
    echo 方法 %%M 的所有训练完成！
    echo ===============================
    echo.
)

echo 所有方法和数据集的训练已完成！
endlocal
pause
