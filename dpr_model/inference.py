import numpy as np
import torch
import argparse
import pandas as pd

from models import Encoder, DPR, binary_cross_entropy_loss
from evaluate import eval_model

from utils import model_save
import warnings
import wandb
from dataloader import parse_data
from tqdm import tqdm
import yaml
from transformers import AutoTokenizer
import faiss
import warnings
import json
import gc
import os

gc.collect()

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config_path", type=str, default="configs/config.yaml",help="config dir")
    args = parser.parse_args()
    return args


def read_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(config)
    return config

def faiss_index(config):
    index = faiss.read_index(config['faiss']['path'])
    print(index.ntotal)
    print("n total ===============")

    return index

def inference(config, data_loader, encoder_tokenizer, index, context_list):
    device = torch.device(config['model']['device'])

    selected_model_dataset = []
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader)):
            question_batch, answer = batch['question'], batch['answer']
            question_encoding = encoder_tokenizer(question_batch, truncation=True, padding=True, max_length=config['hyper_params']['tokenizer_max_length'], return_tensors = 'pt').to(device)

            question_encoder = Encoder(config, device)
            question_encoder.load_state_dict(torch.load(config['model']['passage_model_path']))
            question_encoder.to(device)
            question_encoder.eval()

            question_cls = question_encoder(question_encoding)

            ## index 추출
            _, indices = index.search(question_cls.cpu().numpy().astype("float32"), 40)

            for key, value in enumerate(zip(indices, answer)):
                labels = []
                for ctx in context_list[value[0]]:
                    if value[1].replace(' ','') in ctx.replace(' ',''):
                        labels.append(1)
                    else:
                        labels.append(0)
                        
                selected_model_dataset.append(
                    {
                        "question": question_batch[key],
                        "answer": answer[key],
                        "passages": context_list[value[0]].tolist(),
                        "labels": labels
                    }
                )      

        return selected_model_dataset


def main(config):
    dtype = config["data"]["dtype"]   
    spath = config["data"]["spath"]
    data_loader = parse_data(config, dtype)
    encoder_tokenizer = AutoTokenizer.from_pretrained(config['model']['text_model_path'])
    index = faiss_index(config)

    context_data = pd.read_parquet(config["faiss"]["context"])

    context_list = np.array(context_data["text"])
    selected_model_dataset = inference(config, data_loader, encoder_tokenizer, index, context_list)

    if not os.path.exists(spath):
        os.makedirs(spath)

    if dtype == "train":
        num = config["data"]["num"]
        with open(f"/home/jisukim/DPR/selection_model/datasets/{spath}/selected_model_{dtype}{num}_dataset.json", "w", encoding='utf-8') as f:
            json.dump(selected_model_dataset, f, ensure_ascii=False, indent=4)
    else:
        with open(f"/home/jisukim/DPR/selection_model/datasets/{spath}/selected_model_{dtype}_dataset.json", "w", encoding='utf-8') as f:
            json.dump(selected_model_dataset, f, ensure_ascii=False, indent=4)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    args = parse_args()
    config = read_config(args.config_path)
    main(config)