
# 预测测试集脚本
python ../sentiment/pos_train.py  \
          --bert_model ./models/ \
          --do_lower_case   \
          --train_file /home/xiaoda99_gmail_com/sentiment/datas/small.train.txt  \
          --eval_file /home/xiaoda99_gmail_com/sentiment/datas/small.dev.txt \
          --predict_file /home/xiaoda99_gmail_com/sentiment/datas/test.xlsx \
          --train_batch_size 30 \
          --eval_batch_size 30 \
          --learning_rate 5e-5 \
          --num_train_epochs 10 \
          --output_dir ./models/sentiment_model  \
          --do_predict \
          --predict_result_file ./results/syn-final-3.csv \
          --gradient_accumulation_steps 1 \
          3>&2 2>&1 1>&3 | tee logs/sentiment_predict.log