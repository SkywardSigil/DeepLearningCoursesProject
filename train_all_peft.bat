@echo off
setlocal enabledelayedexpansion

:: �������ݼ��б����� BitFit��
set DATASETS=HumanEval APPS CodeXGLUE_CodeCompletion_token
@REM set DATASETS=HumanEval
:: ����PEFT�����б������� BitFit��
set METHODS=LoRA PromptTuning PrefixTuning IA3
@REM set METHODS=IA3

:: ����ģ��·��
set MODEL_PATH=models/gpt2_local

:: ����Ԥ��������·��
set PREPROCESSED_PATH=results/preprocessed

:: ����CSV����ļ�·��
set CSV_PATH=results/experiment_results.csv

:: ȷ��CSV�ļ��б�ͷ
if not exist "!CSV_PATH!" (
    echo method,epoch,split,loss,perplexity > "!CSV_PATH!"
)

:: ����ÿ��PEFT����
for %%M in (%METHODS%) do (
    echo ===============================
    echo �������з��� %%M ...
    echo ===============================
    
    :: ����ÿ�����ݼ�
    for %%D in (%DATASETS%) do (
        echo ѵ������ %%M �����ݼ� %%D ��...
        
        :: �������ģ��·������־Ŀ¼
        set OUTPUT_MODEL_PATH=results/%%M_finetuned_models/%%D
        set LOG_DIR=results/%%M_logs/%%D
        
        :: �������ģ��·������־Ŀ¼����������ڣ�
        mkdir "!OUTPUT_MODEL_PATH!" 2>nul
        mkdir "!LOG_DIR!" 2>nul
        
        :: ���ݷ��������ض�����������ѵ��
        if "%%M"=="LoRA" (
            python train_peft_gpt2.py ^
                --method LoRA ^
                --num_epochs 15 ^
                --batch_size 16 ^
                --accumulation_steps 2 ^
                --r 4 ^
                --lora_alpha 32 ^
                --lora_dropout 0.1 ^
                --output_model_path "!OUTPUT_MODEL_PATH!" ^
                --log_dir "!LOG_DIR!" ^
                --preprocessed_data_path "!PREPROCESSED_PATH!/%%D" ^
                --csv_path "!CSV_PATH!" ^
                --method_name LoRA ^
                --model_name "!MODEL_PATH!"
        ) else if "%%M"=="PromptTuning" (
            python train_peft_gpt2.py ^
                --method PromptTuning ^
                --num_epochs 15 ^
                --batch_size 16 ^
                --accumulation_steps 2 ^
                --num_virtual_tokens 10 ^
                --output_model_path "!OUTPUT_MODEL_PATH!" ^
                --log_dir "!LOG_DIR!" ^
                --preprocessed_data_path "!PREPROCESSED_PATH!/%%D" ^
                --csv_path "!CSV_PATH!" ^
                --method_name PromptTuning ^
                --model_name "!MODEL_PATH!"
        ) else if "%%M"=="PrefixTuning" (
            python train_peft_gpt2.py ^
                --method PrefixTuning ^
                --num_epochs 15 ^
                --batch_size 16 ^
                --accumulation_steps 2 ^
                --num_prefix_tokens 10 ^
                --output_model_path "!OUTPUT_MODEL_PATH!" ^
                --log_dir "!LOG_DIR!" ^
                --preprocessed_data_path "!PREPROCESSED_PATH!/%%D" ^
                --csv_path "!CSV_PATH!" ^
                --method_name PrefixTuning ^
                --model_name "!MODEL_PATH!"
        ) else if "%%M"=="IA3" (
            python train_peft_gpt2.py ^
                --method IA3 ^
                --num_epochs 15 ^
                --batch_size 16 ^
                --accumulation_steps 2 ^
                --ia3_target_modules attn.c_proj mlp.c_fc mlp.c_proj ^
                --ia3_feedforward_modules mlp.c_fc mlp.c_proj ^
                --output_model_path "!OUTPUT_MODEL_PATH!" ^
                --log_dir "!LOG_DIR!" ^
                --preprocessed_data_path "!PREPROCESSED_PATH!/%%D" ^
                --csv_path "!CSV_PATH!" ^
                --method_name IA3 ^
                --model_name "!MODEL_PATH!"
        )
        
        echo ���� %%M �����ݼ� %%D �ϵ�ѵ����ɡ�
        echo.
    )
    
    echo ===============================
    echo ���� %%M ������ѵ����ɣ�
    echo ===============================
    echo.
)

echo ���з��������ݼ���ѵ������ɣ�
endlocal
pause
