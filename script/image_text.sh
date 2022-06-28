CUDA_VISIBLE_DEVICES=1 python sentiment/image_text_train.py  \
          --bert_model /nas/pretrain-bert/pretrain-pytorch/bert-base-uncased \
          --do_lower_case   \
          --train_file /nas/lishengping/jupyter/external_projects/图片文本分类/image/train.xlsx \
          --eval_file /nas/lishengping/jupyter/external_projects/图片文本分类/image/valid.xlsx \
          --num_train_epochs 6 \
          --train_batch_size 4 \
          --eval_batch_size 4 \
          --learning_rate 3e-5 \
          --output_dir ./output_image \
          --do_train \
          --gradient_accumulation_steps 1 \
          3>&2 2>&1 1>&3 | tee image_text_classifier.log
