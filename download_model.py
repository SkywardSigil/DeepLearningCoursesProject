from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "microsoft/codebert-base"

# 下载并保存分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained("models/codebert-base")

# 下载并保存模型
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.save_pretrained("models/codebert-base")
