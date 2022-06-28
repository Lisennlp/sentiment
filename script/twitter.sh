CUDA_VISIBLE_DEVICES=0 python sentiment/twitter_train.py  \
          --bert_model /nas/pretrain-bert/pretrain-pytorch/bert-base-uncased/ \
          --do_lower_case   \
          --train_file /nas/lishengping/temp/twitter_test.json \
          --epoches 6 \
          --train_batch_size 60 \
          --eval_batch_size 30 \
          --learning_rate 1e-5 \
          --output_dir ./output  \
          --do_train \
          --gradient_accumulation_steps 1 \
          3>&2 2>&1 1>&3 | tee twitter_train2.log

