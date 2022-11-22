#import torch
from transformers import AutoModel, AutoTokenizer
ModelName = 'kogpt2'
tokenizer = AutoTokenizer.from_pretrained(ModelName)
model = AutoModel.from_pretrained(ModelName)
# 이럴 경우 bert모델임