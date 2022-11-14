import Trainer
import aboutDataSets
import numpy as np
import pandas as pd
import torch
import re  # 정규식 계산
import urllib.request  # url로 csv파일 받아오기
from pytorch_lightning import Trainer  # pip install pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW  # optimizer
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        a = ""
        while 1:
            input_ids = torch.LongTensor(Trainer.tokenizer.encode(Trainer.Q_TKN + q + Trainer.SENT + Trainer.A_TKN + a)).unsqueeze(dim=0) # 이 sent뭐임??
            pred = Trainer.model(input_ids)
            pred = pred.logits
            gen = Trainer.tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == Trainer.EOS:
                break
            a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))
#