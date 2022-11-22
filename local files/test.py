import Trainer
import torch

# # voice recognition -> only tts == == == ==
# import speech_recognition as sr
# import os
from gtts import gTTS
import playsound

__author__ = "baeksh0330@gachon.ac.kr"

def reply(text, a):  #a=fileNum Count
    tts = gTTS(text=text, lang='ko')
    fileName = text+str(a)+'.mp3'
    tts.save(fileName)
    playsound.playsound(fileName)  # 파일 읽어주기
    playsound.playsound(text)

# chat here!
with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        answer = ""
        while 1:
            input_ids = torch.LongTensor(
                Trainer.tokenizer.encode(Trainer.Q_TKN + q + Trainer.SENT + Trainer.A_TKN + answer)).unsqueeze(dim=0) # 이 sent뭐임??
            pred = Trainer.model(input_ids)
            pred = pred.logits
            gen = Trainer.tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == Trainer.EOS:
                break
            answer += gen.replace("▁", " ")
        #reply(answer, 1) # => 지금 한국어-영어 파일명 충돌이 일어남
        print("Chatbot > {}".format(answer.strip()))
