nohup python3 train.py --config_path ./config/train_batch_32.yaml > ./logs/train_output_gold32.log &
nohup python3 train.py --config_path ./config/train_batch_16.yaml > ./logs/train_output_gold16.log &
nohup python3 train.py --config_path ./config/train_batch_8.yaml > ./logs/train_output_gold8.log &
nohup python3 train.py --config_path ./config/train_batch_4.yaml > ./logs/train_output_gold4.log &
