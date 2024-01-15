import json
import torch

# 모델 파일 경로
# model_path = '/home/jisukim/DPR/dpr_model/results/best_model_gold32_3.pt'
# model_path = '/ml_data/retrieval_model/results/best_model_gold16_3.pt'
# model_path = '/ml_data/retrieval_model/results/best_model_gold8_5.pt'
model_path = '/ml_data/retrieval_model/results/best_model_gold4_5.pt'

# 모델 로드
model = torch.load(model_path, map_location=torch.device('cpu'))

from collections import OrderedDict

ordered_dict_question = OrderedDict()
ordered_dict_passage = OrderedDict()

for key, value in model.items():
    if key.split('.')[0] == "question_encoder":
        ## load하기 위해 encoder로 rename
        key = '.'.join(key.split('.')[1:])
        ordered_dict_question[key] = value
    if key.split('.')[0] == "passage_encoder":
        key = '.'.join(key.split('.')[1:])
        ordered_dict_passage[key] = value

# torch.save(ordered_dict_question, "/home/jisukim/DPR/dpr_model/results/question_gold_32.pt")
# torch.save(ordered_dict_passage, "/home/jisukim/DPR/dpr_model/results/passage_gold_32.pt")

# torch.save(ordered_dict_question, "/home/jisukim/DPR/dpr_model/results/question_gold_16.pt")
# torch.save(ordered_dict_passage, "/home/jisukim/DPR/dpr_model/results/passage_gold_16.pt")

# torch.save(ordered_dict_question, "/home/jisukim/DPR/dpr_model/results/question_gold_8.pt")
# torch.save(ordered_dict_passage, "/home/jisukim/DPR/dpr_model/results/passage_gold_8.pt")

torch.save(ordered_dict_question, "/home/jisukim/DPR/dpr_model/results/question_gold_4.pt")
torch.save(ordered_dict_passage, "/home/jisukim/DPR/dpr_model/results/passage_gold_4.pt")
