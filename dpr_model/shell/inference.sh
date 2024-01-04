# nohup python3 /home/jisukim/DPR/dpr_model/inference.py --config_path /home/jisukim/DPR/dpr_model/config/inference_batch_32_test.yaml > /home/jisukim/DPR/dpr_model/logs/inference_batch_32.log &
# nohup python3 /home/jisukim/DPR/dpr_model/inference.py --config_path /home/jisukim/DPR/dpr_model/config/inference_batch_32_train1.yaml > /home/jisukim/DPR/dpr_model/logs/inference_batch_32_train1.log &
## 아래부터 돌려야함
nohup python3 /home/jisukim/DPR/dpr_model/inference.py --config_path /home/jisukim/DPR/dpr_model/config/inference_batch_32_train2.yaml > /home/jisukim/DPR/dpr_model/logs/inference_batch_32_train2.log &
nohup python3 /home/jisukim/DPR/dpr_model/inference.py --config_path /home/jisukim/DPR/dpr_model/config/inference_batch_32_train3.yaml > /home/jisukim/DPR/dpr_model/logs/inference_batch_32_train3.log &
nohup python3 /home/jisukim/DPR/dpr_model/inference.py --config_path /home/jisukim/DPR/dpr_model/config/inference_batch_32_valid.yaml > /home/jisukim/DPR/dpr_model/logs/inference_batch_32_valid.log &
