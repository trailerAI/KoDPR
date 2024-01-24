import numpy as np
import torch
import pathlib
import argparse

from models import negative_log_loss

from dpr_dataloader import parse_data


#
def main(opt):
    save_model_path, dataset_path, context_path, device = opt.model_path, opt.dataset, opt.context, opt.device
    # model load
    model = torch.jit.load(str(pathlib.Path(save_model_path)))
    model.to(device)

    test_data_loader = parse_data(dataset_path, context_path)

    test_loss, test_acc = eval_model(model, test_data_loader, negative_log_loss)

    print(f'Evaluation: Test Loss:{test_loss}, Test Accuracy:{test_acc}')


# eval
def eval_model(model, data_loader, optimizer, loss_fn, device, config, encoder_tokenizer):
    losses = []
    accuracy = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            question_batch, positive_passage_batch  = batch['question'], batch['positive_passage']#, batch['bm_25']
            # question_batch, positive_passage_batch, bm25_passage_batch  = batch['question'], batch['positive_passage'], batch['bm_25']

            # if config['data']['bm_25'] == True:
            #     passage_encoding = encoder_tokenizer(positive_passage_batch+bm25_passage_batch, truncation=True, padding=True, max_length=config['hyper_params']['tokenizer_max_length'], return_tensors = 'pt').to(device)

            # else:
            passage_encoding = encoder_tokenizer(positive_passage_batch, truncation=True, padding=True, max_length=config['hyper_params']['tokenizer_max_length'], return_tensors = 'pt').to(device)

            question_encoding = encoder_tokenizer(question_batch, truncation=True, padding=True, max_length=105, return_tensors = 'pt').to(device)
            

            questions_cls, passages_cls = model(question_encoding, passage_encoding)    
                
            optimizer.zero_grad()

            loss, correct_count = loss_fn(questions_cls, passages_cls, device)
            losses.append(loss.item())
            accuracy.append(correct_count/config['hyper_params']['batch_size'])

    return np.mean(losses), np.mean(accuracy)


# 
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dataset')    
    parser.add_argument('--context', type=str, default='dataset')    
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    opt = parser.parse_args()
    return opt

#
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
