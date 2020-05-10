#/home/xiaoda99_gmail_com/chinese_roberta_wwm_ext_pytorch/
python ../sentiment/pos_train.py  \
          --bert_model /home/xiaoda99_gmail_com/chinese_roberta_wwm_ext_pytorch/ \
          --do_lower_case   \
          --train_file /home/xiaoda99_gmail_com/sentiment/datas/small.train.txt  \
          --eval_file /home/xiaoda99_gmail_com/sentiment/datas/small.dev.txt \
          --train_batch_size 30 \
          --eval_batch_size 30 \
          --learning_rate 1e-5 \
          --num_train_epochs 10 \
          --output_dir ./models/sentiment_model5  \
          --do_train \
          --gradient_accumulation_steps 1 \
          3>&2 2>&1 1>&3 | tee logs/sentiment_train.log

