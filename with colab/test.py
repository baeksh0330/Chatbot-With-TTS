__author__ = "baeksh0330@gachon.ac.kr"

import torch
import Trainer
from gtts import gTTS
from IPython.display import Audio

def reply(text, a):  #a=fileNum Count
    tts = gTTS(text=text, lang='ko')
    fileName = text+str(a)+'.mp3'
    tts.save(fileName)
    respond = Audio(fileName, autoplay=True)
    display(respond)


# chat here!
with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "잘 가":
            reply("다음에 또 만나요.", 1)
            break
        answer = ""
        while 1:
            input_ids = torch.LongTensor(Trainer.tokenizer.encode(Trainer.Q_TKN + q + Trainer.SENT + Trainer.A_TKN + answer)).unsqueeze(dim=0).to('cuda:0')
            pred = Trainer.model(input_ids)
            pred = pred.logits
            gen = Trainer.tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
            if gen == Trainer.EOS:
                break
            answer += gen.replace("▁", " ")
        reply(answer, 1)
        print("Chatbot > {}".format(answer.strip()))
