import math
import numpy as np
import pandas as pd
import random  # 챗봇의 랜덤 기능
import re  # 정규식 연산
import torch
import urllib.request  # 챗봇 데이터 다운로드 위함
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel  # model은 GPT2, optimizer는 adamW
from transformers import PreTrainedTokenizerFast  # tokenizer는 PretrainedTokenizerFast
import tokenizers

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
    filename="ChatBotData.csv",)  # 챗봇 데이터 다운로드 : Q, A, Label로 이루어져 있다.
# 감정 레이블은 label에 정의된 일상다반사 0, 이별(부정) 1, 사랑(긍정) 2)를 그대로 적용

ChatData = pd.read_csv("ChatBotData.csv")
ChatData = ChatData[:300]  # test용으로 300개만 처리 : reference
# print(ChatData.head())  # pandas에서 불러온 상위데이터 확인하기
# print(ChatData.info())
# print(ChatData.isnull().sum()) # 불필요한 NULL값 확인.

# 구두점 처리 : 이 과정이 꼭 필요한지는 모르겠다. 아래에서 다시 진행하므로 일단 주석처리. 참고용.
# questions = []
# for sentence in ChatData['Q']: # 질문 부분 전처리
#     sentence = re.sub(r"([?.!,])", r" \1",sentence)
#     sentence = sentence.strip()
#     questions.append(sentence)
#
# answers = []
# for sentence in ChatData['A']:  # 답변변 부분 전처리
#     sentence = re.sub(r"([?.!,])", r" \1",sentence)
#     sentence = sentence.strip()
#     answers.append(sentence)
# print(questions[:5])
# print(answers[:5])

# tokens
Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
SENT = '<unused1>' # 미사용토큰을 정의해 필요한 태스크에 따라 자유롭게 정의하도록.
MASK = "<mask>"  # <ununsed0> ?

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",  # 언어 모델. kogpt2를 쓸 것.
    bos_token=BOS,  # 문장의 시작
    eos_token=EOS,  # 문장의 끝
    unk_token="<unk>",  # 모르는 단어를 나타내는 token
    pad_token=PAD,  # 동일한 batch 안에서 입력의 크기를 동일하게 만들기 위한 padding
    mask_token=MASK, )  # attention mask

# encoded = koGPT2_TOKENIZER("This is Test for tokenizer performance")
# print("Test Encoding: ", encoded)
# encoding3 = tokenizer("넌 당연하지 않아")
# print("Encoding result = ", encoding3)

# for ts in encoding3:
#     print('{}---->{}'.format(ts, koGPT2_TOKENIZER([ts]))) # tokenizer decode함수를 못찾겠다. 디코딩 어떻게 함.

# 챗봇 데이터 처리
class ChatDataset(Dataset):
    def __init__(self, chats, max_len=40):  # 데이터셋 전처리
        self.data = chats
        self.max_len = max_len  # 문장의 최대 길이는 40으로 고정(padding이용)
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.sent_eos = EOS
        self. mask = MASK
        self.tokenizer = tokenizer  # 모델 tokenizer

    def __len__(self):  # 길이 반환
        return len(self.data)

    def __getitem__(self, idx):  # load한 데이터를 DataLoader로 넘겨주는 메서드
        turn = self.data.iloc[idx]  # idx 데이터
        q = turn["Q"]  # 질문을 가져오기
        q = re.sub(r"([?.!,])", r" ", q)  # 질문 : 구둣점 제거

        a = turn["A"]
        a = re.sub(r"([?.!,])", r" ", a)  # 답변 : 구둣점 제거

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.sent_eos)  # eos: 문장의 끝 token
        a_len = len(a_toked)

        if q_len > self.max_len:  # 질문의 길이가 최대 길이 보다 크면
            a_len = self.max_len - q_len  # 질문이 너무 길면 답변의 길이도 줄여 주는 건가?
            if a_len <= 0:  # 답변의 길이가 너무 긴 경우 (확인용 인가봄)
                q_toked = q_toked[-(int(self.max_len / 2)):]  # 질문 길이를 최대 길이의 반으로
                q_len = len(q_toked)
                a_len = self.max_len - q_len  # 답변의 길이를 최대 - 질문 길이

            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len / 2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        # 답변 labels = [mask, mask, ...., mask, ..., <bos>,..답변.. <eos>, <pad>....]
        labels = [self.mask, ] * q_len + a_toked[1:]

        # mask = 질문 길이 0 + 답변 길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)

        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        # token string 또는 token string의 리스트를 token id 또는 Token id의 리스트로 변환한다.
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)  # 합치는 이유는?

        # 기준 최대길이보다 token ids 길이가 작을 경우 -> padding으로 채워주기
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        # 질문+답변, 마스크, 답변
        return (token_ids, np.array(mask), labels_ids)

# 배치 데이터를 만들기 위한 collate_batch함수
def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

# dataset / dataloader 정의
train_set = ChatDataset(ChatData, max_len=40)

#윈도우 환경에서 num_workers 는 무조건 0으로 지정, 리눅스에서는 2
# 여기서 데이터 튜닝 :
train_dataloader = DataLoader(train_set,
                              batch_size=32,
                              num_workers=0,
                              shuffle=True,
                              collate_fn=collate_batch,) # 뭐지?

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# hyper-parameters - fine_tuning
# D_model = 256
# numLayers = 2
# numHeads = 8
# dff = 512 # 이거 뭐임
# dropOut = 0.1
#
# lr = 0.01
# epochs = 10
# optimizer = AdamW(lr, eps=1e-6)



# def __main__():
#     print("start")
#     for batch_idx, samples in enumerate(train_dataloader):
#         token_ids, mask, label = samples
#         print("token_ids ====> ", token_ids)
#         print("mask =====> ", mask)
#         print("label =====> ", label)
#     print("end")