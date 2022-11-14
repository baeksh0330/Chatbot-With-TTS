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

# 토큰은 변경 가능. unused는 임의로 변경가능.
Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token=BOS,
                                                    eos_token=BOS,
                                                    unk_token='unk',
                                                    pad_token=PAD,
                                                    mask_token=MASK)
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
    filename="ChatBotDataMain.csv",
)

ChatData = pd.read_csv("ChatBotDataMain.csv")
ChatData = ChatData[:300]
# print(ChatData.head())

#dataset 만들기
dataset = aboutDataSets.ChatDataset(ChatData)

batch_size = 32
num_workers = 0

def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)
# 아래 collate_batch 변수때문에 여기 한번 더 호출.

#dataloader 선언
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = aboutDataSets.ChatDataset(ChatData, max_len=40)
train_dataLoader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              collate_fn=collate_batch,) # 예제는 그냥 collate_batch인데 이게 뭐지. 함수를 이렇게 호출할 수가 있던가....

model.to(device)  # cpu에서 돌림
model.train()
lr = 3e-5
criterion = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epoch = 10
sneg = -1e18 # 이 변수는 뭐야?

# for b, s in enumerate(train_dataLoader): # 여기서 오류가 남
#     print(s)


# 학습 시작
print("::start::")
for epoch in range(epoch):
    for batch_idx, samples in enumerate(train_dataLoader):
        print(batch_idx, samples)
        optimizer.zero_grad() # 초기화 느낌
        token_ids, mask, label = samples
        out = model(token_ids)
        out = out.logits  # returns a new tensor with the logit of the elements of input
        #여기 함수 공부 할것. upsqueeze는 차원 올리기인데.
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)
        avg_loss = loss.sum() / mask.sum() # avg_loss[0] / avg_loss[1] <- loss 정규화
        avg_loss.backward()
        # 학습 끝
        optimizer.step()
print("end")

# with torch.no_grad():
#     while 1:
#         q = input("user > ").strip()
#         if q == "quit":
#             break
#         a = ""
#         while 1:
#             input_ids = torch.LongTensor(tokenizer.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0) # 이 sent뭐임??
#             pred = model(input_ids)
#             pred = pred.logits
#             gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
#             if gen == EOS:
#                 break
#             a += gen.replace("▁", " ")
#         print("Chatbot > {}".format(a.strip()))
# #