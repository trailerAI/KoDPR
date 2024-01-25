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
데이터는 [ai-hub데이터와 위키피디아 데이터]를 활용했습니다. 


## Train
DPR 모델 학습을 진행하기 위해 train_batch_{N}.yaml 설정을 수정한 다음, `shell/train.sh`를 실행하면 됩니다.

```
chmod 775 ./shell/train.sh
./train.sh
```

## Split Models
Faiss index를 생성하고 inference 하기 위해, 학습한 모델을 question, passage모델로 분할합니다.
```
python model_split.py --model_path ./results/best_model_gold32_3.pt --question_path ./results/question_gold_32.pt --passage_path ./results/passage_gold_32.pt
```

## Faiss Index 생성
Faiss Index를 생성하기 위해 faiss_batch_32_{N}.yaml 설정을 수정한 다음, `shell/faiss.sh`를 실행하면 됩니다.

```
chmod 775 ./shell/faiss.sh
./faiss.sh
```

## Inference
Inference결과를 확인하기 위해 inference_batch_{N}_test.yaml 설정을 수정한 다음, `shell/inference.sh`를 실행하면 됩니다.

```
chmod 775 ./shell/inference.sh
./inference.sh
```

이때, [**Selection Model**](https://github.com/trailerAI/SelectionModel)을 학습시키기 위한 데이터 셋을 생성할 경우, inference_batch_32_train.yaml, inference_batch_32_valid.yaml, inference_batch_32_valid.yaml 설정을 수정한 다음, `shell/inference.sh`를 실행하면 됩니다.


## Results
| N  | Top@1 | Top@5 | Top@10 | Top@20 |
|----|-------|-------|--------|--------|
| 16 | 37.26%| 61.16%| 70.05% | 78.15% |
| 32 | 37.87%| 61.81%| 71.04% | 79.04% |


## Contributors
[TakSung Heo](https://github.com/HeoTaksung), [Jisu, Kim](https://github.com/merry555), [Juhwan, Lee](https://github.com/juhwanlee-diquest),  and [Minsu Jeong](https://github.com/skaeads12)


## License
Apache License 2.0 lisence
...
