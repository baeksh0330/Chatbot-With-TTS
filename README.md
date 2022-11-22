# Chatbot-With-TTS
<h3>2022 toy project 2</h3>
<h4>Chatbot with voice synthesis/generation using GTTS module</h4>

* LM = KoGPT2 / GPT2LMHeadModel
* tokenizer = PreTrainedTokenizerFast
<br>

<h2>Requirements</h2>

* python 3.7.15
* transformers
* GTTS
* Google colab / pycharm(local)
* Torch
* ... more in requirements.txt
<br><br>

<h2>reference</h2>

* Google TTS : https://gtts.readthedocs.io/en/latest/module.html <br>
* Basic Python Chatbot model - KoGPT2 : https://github.com/SKT-AI/KoGPT <br>
* Korean conversation dataset : https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv

##### git clone
```python
!git clone https://github.com/baeksh0330/Chatbot-With-TTS
%cd /content/Chatbot-With-TTS/with colab
!pip3 install --upgrade pip
!pip3 install -r requirements.txt 
%run test.py  # main chat here
```
* first execution : model train with dataset is required. <br>It could take time(20mins for almost ten thousands data)- to reduce execute time, reduce dataset's size: modify dataset[n::m] in trainer.py
<br>
<h2>Result</h2>

![image](https://user-images.githubusercontent.com/78344141/203339908-4d56a497-8bec-4dd5-bcc1-3c7f10517c69.png)
<h2>Quit condition</h2>
> chat '잘 가'
