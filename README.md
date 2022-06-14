# English-Korean transliterator (영-한 음차 변환기)
이 프로젝트는 영어 단어를 한글 발음 표기로 변환하는 프로그램입니다. (e.g. ```transformer``` &rarr; ```트랜스포머```)  
<br>
[허깅페이스](https://huggingface.co/models?pipeline_tag=text2text-generation&sort=downloads)의 ```Text2Text Generation``` Task 의 사전학습 언어 모델을 사용하였습니다.

```Fine-tuning``` 한 음차 변환 모델은 [여기](https://huggingface.co/eunsour/en-ko-transliterator)에서 확인하실 수 있습니다.

- Use pre-trained model from [Google-mT5 : Multilingual T5](https://huggingface.co/google/mt5-base)

<br>

## Prerequisites
```
$ pip install -r requirements.txt
```
`CUDA` 버전에 맞는 `torch` 의 설치가 필요합니다.
```
# CUDA 11.3
$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

<br>

## Usage
### ver1. Train model
```bash
$ python3 transliteration.py --train
('>> total number of data:', 56699)
('>> number of train data:', 51029)
('>> number of test data:', 5670)
('> Preprocessing',)
('> Preprocessing',)
('> Train Model Start...',)
INFO:simpletransformers.t5.t5_model: Training started
```
<br>

### ver2. Use pre-trained language model
```bash
$ python3 transliteration.py --test
('> Pretrained Model Start...',)
Generating outputs: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 10.93it/s]
Decoding outputs: 100%|██████████████████████████████████████| 4/4 [00:00<00:00,  6.20it/s]
machinelearning :       머신러닝
deeplearning    :       딥러닝
transformer     :       트랜스포머
attention       :       어텐션
```


```bash
$ python3 transliteration.py --decode
종료는 'q' 입니다.
>> transformer
('> Pretrained Model Start...',)
Generating outputs: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 10.81it/s]
Decoding outputs: 100%|██████████████████████████████████████| 1/1 [00:00<00:00,  1.47it/s]
트랜스포머
>> kakao
Generating outputs: 100%|██████████████████████████████████████| 1/1 [00:00<00:00, 9.71it/s]
Decoding outputs: 100%|██████████████████████████████████████| 1/1 [00:00<00:00,  5.78it/s]
카카오
>>
```
<br>

## Run Gradio app
```bash
$ python transliteration.py --gradio
Running on local URL:  http://127.0.0.1:7860/

To create a public link, set `share=True` in `launch()`.
```
<br>

## References
- 데이터셋은 muik의 [transliteration](https://github.com/muik/transliteration) 데이터를 사용하였습니다. 
- 모델의 소스코드 부분은 grit-mind의 [engkor_transliterator](https://github.com/gritmind/engkor_transliterator) 를 참고하였습니다.
