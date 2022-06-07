# engtokor-transliterator
Transliteration converter from English to Korean.

<br>

## Prerequisites
```
$ python3 -m venv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```
`CUDA` 버전에 맞는 `torch` 의 설치가 필요합니다.
```
# CUDA 11.3
$ pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```


## Usage
### ver1. Train model
```
$ python3 transliteration.py --train
```

### ver2. Pretrained Language model
```
$ python3 transliteration.py --test
```

## References
- 데이터셋은 muik의 [transliteration](https://github.com/muik/transliteration) 데이터를 사용하였습니다. 
- 모델의 소스코드 부분은 grit-mind의 [engkor_transliterator](https://github.com/gritmind/engkor_transliterator) 를 참고하였습니다.
