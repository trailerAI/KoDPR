# KoDPR
klue/roberta-base 모델을 활용한 한국어 DPR 모델 입니다.

(본 실험은 [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) 논문을 기반으로 수행되었습니다.)

## Setting
```
cd KoDPR
poetry shell
poetry install
```


## Data
데이터는 [ai-hub데이터와 위키피디아 데이터](https://drive.google.com/drive/folders/1Vs4pTehFCmPNgak3MxhRHbyuIGN-hCSx?usp=sharing)를 활용했습니다. 

- train_preprocess.json: Ai-hub 데이터 셋을 정제한 학습 데이터 입니다.
- valid_preprocess.json: Ai-hub 데이터 셋을 정제한 검증 데이터 입니다.
- test_preprocess.json: Ai-hub 데이터 셋을 정제한 테스트 데이터 입니다.
- context.json: Ai-hub 데이터 셋 중 context만 추출한 데이터 입니다.
- context_total.parquet: FAISS 인덱스를 만들기 위해 위키피디아 데이터셋과 ai-hub의 context 데이터 셋을 klue/roberta-base를 활용해 토크나이징한 다음 512 길이로 chunking해서 저장한 파일 입니다.


## KoDPR Model Train
```
python train.py --config_path ./config/train_batch_4.yaml
```

학습한 모델을 question, passage모델로 분할하고 싶을 경우 model_split.py를 참고해 실행합니다.


## Faiss Index 생성
```
python gen_db.py --config_path ./config/faiss_batch_4.yaml
```

## Inference
Inference결과 top k의 accuracy 결과를 확인할 수 있습니다.

```
python inference.py --config_path ./config/inference_batch_4_test.yaml
```

## Results
| Batch size  | Top@1 | Top@5 | Top@10 | Top@20 |
|----|-------|-------|--------|--------|
| 4  |       |       |        |        |
| 8  |       |       |        |        |
| 16 |       |       |        |        |
| 32 |       |       |        |        |


## Contributors
[Jisu, Kim](https://github.com/merry555), [Juhwan, Lee](https://github.com/juhwanlee-diquest), [TakSung Heo](https://github.com/HeoTaksung), and [Minsu Jeong](https://github.com/skaeads12)


## License
Apache License 2.0 lisence
