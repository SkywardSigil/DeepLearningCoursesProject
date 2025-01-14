from datasets import load_from_disk

# 查看HumanEval数据集
humaneval = load_from_disk("results/preprocessed/HumanEval")
print(humaneval["train"][0])

# 查看APPS数据集
apps = load_from_disk("results/preprocessed/APPS")
print(apps["train"][0])

# 查看CodeXGLUE CodeCompletion-token数据集
codexglue = load_from_disk("results/preprocessed/CodeXGLUE_CodeCompletion_token")
print(codexglue["train"][0])
