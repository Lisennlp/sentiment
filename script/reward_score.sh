CUDA_VISIBLE_DEVICES=1 python sentiment/reward_score.py  \
          --bert_model /nas2/archived/qsj/bert-model/bert-base-chinese \
          --do_lower_case   \
          --train_file /nas/lishengping/datas/transformer-xl/reward_score/final/train.txt \
          --eval_file /nas/lishengping/datas/transformer-xl/reward_score/final/valid.txt \
          --num_train_epochs 10 \
          --train_batch_size 4 \
          --eval_batch_size 8 \
          --learning_rate 3e-5 \
          --output_dir ./output3 \
          --do_train \
          --gradient_accumulation_steps 1 \
          3>&2 2>&1 1>&3 | tee reward_score.log
