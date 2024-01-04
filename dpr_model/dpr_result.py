import json
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import argparse


def retrieval_performance(data_length, labels):
    count_dict = {i+1:0 for i in range(len(labels[0]))}
    for i in range(len(labels)):
        if 1 in labels[i]:
            idx = labels[i].index(1)
            count_dict[idx+1] += 1

    acc_count = 0
    acc_dict = {}

    for k, v in zip(count_dict.keys(), count_dict.values()):
        acc_count += v
        acc_dict[k] = acc_count/data_length
        
    return acc_dict


def gen_dataset(fpath, dtype):
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

    if dtype == "train":
        with open(f"/home/jisukim/DPR/selection_model/datasets/{fpath}/selected_model_{dtype}1_dataset.json") as f:
            data1 = json.load(f)

        with open(f"/home/jisukim/DPR/selection_model/datasets/{fpath}/selected_model_{dtype}1_dataset.json") as f:
            data2 = json.load(f)

        with open(f"/home/jisukim/DPR/selection_model/datasets/{fpath}/selected_model_{dtype}1_dataset.json") as f:
            data3 = json.load(f)

        data = data1 + data2 + data3

    else:
        with open(f"/home/jisukim/DPR/selection_model/datasets/{fpath}/selected_model_{dtype}_dataset.json") as f:
            data = json.load(f)


    if dtype == "test":
        test_labels = []
        for i in range(len(data)):
            test_labels.append(data[i]['labels'])
            
        test_acc = retrieval_performance(len(test_labels), test_labels)
        for i in [1, 5, 10, 15, 20]:
            print("Test Accuracy ===============")
            print(i, ':', test_acc[i])


def main(fpath, dtype):
    gen_dataset(fpath, dtype)


def parse_args():
    parser = argparse.ArgumentParser(description="Run script")
    parser.add_argument('-fpath', '--fpath', type=str, help='fpath', required=True)
    parser.add_argument('-dtype', '--dtype', type=str, help='dtype', required=True)

    args = parser.parse_args()
    return args.fpath, args.dtype

if __name__ == '__main__':
    main(*parse_args())