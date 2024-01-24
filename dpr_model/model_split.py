import torch
import argparse

def main(model_path, question_path, passage_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))

    from collections import OrderedDict

    ordered_dict_question = OrderedDict()
    ordered_dict_passage = OrderedDict()

    for key, value in model.items():
        if key.split('.')[0] == "question_encoder":
            key = '.'.join(key.split('.')[1:])
            ordered_dict_question[key] = value
        if key.split('.')[0] == "passage_encoder":
            key = '.'.join(key.split('.')[1:])
            ordered_dict_passage[key] = value

    torch.save(ordered_dict_question, question_path)
    torch.save(ordered_dict_passage, passage_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run script")
    parser.add_argument('-model_path', '--model_path', type=str, help='DPR model path', required=True)
    parser.add_argument('-question_path', '--question_path', type=str, help='Question model path', required=True)
    parser.add_argument('-passage_path', '--passage_path', type=str, help='Passage model path', required=True)

    args = parser.parse_args()
    return args.model_path, args.question_path, args.passage_path


if __name__ == '__main__':
    main(*parse_args())
