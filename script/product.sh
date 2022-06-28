CUDA_VISIBLE_DEVICES=1 python sentiment/product_train.py  \
          --bert_model /nas/pretrain-bert/pretrain-pytorch/bert-base-chinese \
          --do_lower_case   \
          --train_file /nas/lishengping/jupyter/external_project/product_classifier/data.xls \
          --num_train_epochs 6 \
          --train_batch_size 24 \
          --eval_batch_size 24 \
          --learning_rate 3e-5 \
          --output_dir ./output2 \
          --do_train \
          --gradient_accumulation_steps 1 \
          3>&2 2>&1 1>&3 | tee product_classifier.log
