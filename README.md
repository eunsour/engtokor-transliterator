# engtokor-transliterator
Transliteration converter from English to Korean.

<br>

## Prerequisites
```
$ python3 -m venv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
```
torch는 CUDA Version에 맞게 설치해주시면 됩니다
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
