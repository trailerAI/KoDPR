# KoDPR
한국어 DPR 모델 입니다.

## 데이터
데이터는 [ai-hub데이터와 위키피디아 데이터](https://drive.google.com/drive/folders/1Vs4pTehFCmPNgak3MxhRHbyuIGN-hCSx?usp=sharing)를 활용했습니다. 

- train_preprocess.json: Ai-hub 데이터 셋을 정제한 학습 데이터 입니다.
- valid_preprocess.json: Ai-hub 데이터 셋을 정제한 검증 데이터 입니다.
- test_preprocess.json: Ai-hub 데이터 셋을 정제한 테스트 데이터 입니다.
- context.json: Ai-hub 데이터 셋 중 context만 추출한 데이터 입니다.
- context_total.parquet: FAISS 인덱스를 만들기 위해 위키피디아 데이터셋과 ai-hub의 context 데이터 셋을 klue/roberta-base를 활용해 토크나이징한 다음 512 길이로 chunking해서 저장한 파일 입니다.
