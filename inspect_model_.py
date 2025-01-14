from transformers import GPT2LMHeadModel

def list_model_modules(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    for name, module in model.named_modules():
        print(name)

if __name__ == "__main__":
    model_path = "E:/NJU/深度学习平台/final/exp_new/models/gpt2_local"  # 请根据实际路径修改
    list_model_modules(model_path)
