CUDA_VISIBLE_DEVICES=1 python sentiment/image_text_train2.py  \
          --bert_model /nas/pretrain-bert/pretrain-pytorch/bert-base-uncased \
          --do_lower_case   \
          --train_file /nas/lishengping/temp/image_text.train \
          --eval_file /nas/lishengping/temp/image_text.dev \
          --num_train_epochs 6 \
          --train_batch_size 10 \
          --eval_batch_size 20 \
          --learning_rate 3e-5 \
          --output_dir ./output_image \
          --do_train \
          --gradient_accumulation_steps 1 \
          3>&2 2>&1 1>&3 | tee image_text_classifier.log
