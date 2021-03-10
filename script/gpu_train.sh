#/home/xiaoda99_gmail_com/chinese_roberta_wwm_ext_pytorch/
CUDA_VISIBLE_DEVICES=2,3 python sentiment/gpu_train.py  \
          --bert_model /nas/pretrain-bert/pretrain-pytorch/bert-base-chinese \
          --do_lower_case   \
          --train_file datas/train.txt  \
          --eval_file datas/dev.txt \
          --predict_file datas/新冠肺炎.xlsx \
          --epoches 6 \
          --train_batch_size 60 \
          --eval_batch_size 30 \
          --learning_rate 1e-5 \
          --num_train_epochs 10 \
          --output_dir ./output  \
          --do_train \
          --do_predict  \
          --gradient_accumulation_steps 1 \
          3>&2 2>&1 1>&3 | tee sentiment_train.log

