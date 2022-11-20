import Trainer
import torch
# import aboutDataSets
# import numpy as np
# import pandas as pd
# import re  # 정규식 계산
# import urllib.request  # url로 csv파일 받아오기
# from pytorch_lightning import Trainer  # pip install pytorch_lightning
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.core.lightning import LightningModule
# from torch.utils.data import DataLoader, Dataset
# from transformers.optimization import AdamW  # optimizer
# from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# # voice recognition -> only tts == == == ==
# import speech_recognition as sr
# import os
from gtts import gTTS
import playsound

__author__ = "baeksh0330@naver.com"

def reply(text, a):  #a=fileNum Count
    tts = gTTS(text=text, lang='ko')
    fileName = text+str(a)+'.mp3'
    tts.save(fileName)  # 저장을 해야 하나? 제외할 수 있으면 제외할 것
    playsound.playsound(fileName)  # 파일 읽어주기

# chat here!
with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        answer = ""
        while 1:
            input_ids = torch.LongTensor(Trainer.tokenizer.encode(Trainer.Q_TKN + q + Trainer.SENT + Trainer.A_TKN + answer)).unsqueeze(dim=0) # 이 sent뭐임??
            pred = Trainer.model(input_ids)
            pred = pred.logits
            gen = Trainer.tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == Trainer.EOS:
                break
            answer += gen.replace("▁", " ")
        reply(answer, 1) # count는 아무 숫자나 해도 상관없으므로(일단은)
        print("Chatbot > {}".format(answer.strip()))
## 현재 문제 : 대화가 너무 부자연스러움!!