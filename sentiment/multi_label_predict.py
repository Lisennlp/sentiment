import os
import logging
import argparse
import random
import json

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch.utils.data import DataLoader

import tokenization
from modeling import BertForSequenceClassification, BertForMultipleChoice


device = 'cuda'
label_map = ['菜品味道', '等待情况',  '就餐环境', '服务情况']

# text = '|上菜很快，最爱的凤爪和虾饺都好吃，在广州有很多分店，不愧是老字号，每家店都很多人'
# text = '|老顾客了。。消费了300多送了两盒初生蛋，盆友过来把招牌都点了，5个人没吃完，价格公道，性价比高。味道依旧～喜欢牛肉球，油条，凤爪，红米肠~'
# text = '|呢间总店，饭点排噻长龙，2人消费104元，4样嘢，1.红米肠外软，里面脆网有点硬，很油，吃两块不想吃了，一般2.椰香脆皮年糕有失水准，一层油，差评3.咸煎饼乳香味不够，炸得不够透，似工厂加工送货返炸，差评4.桑叶还可以，有少少味精味服务礼貌几好，中间有人加水'
# text = '|广州酒家是老品牌了，粤菜都是做得很不错~环境服务都很好~就是贵了点'
# text  = '|去的时候快十点了，客服也没有不耐烦，谢谢'
# text = '|今天周六中午，预计周末人多专程提前在11点半前到达，意外地不必等位就让我们上三楼桌子。口味：一如往常的味道，各款茶点都挺好吃??凭良心说这附近周边茶楼就这家的茶点最合心意！当然价格不算美好，尚能接受。最是喜欢这儿的冰榴莲，入口香味浓郁，口齿留香欲罢不能。鸡汁四色虾饺内馅是虾肉很足料，塞得满满当当的，虾肉爽口有姜汁味。牛肉拉肠的肠粉柔滑细腻而牛肉嫩滑。新品千叶豆腐正搞特价，看那买相就刺激食欲，配了小半个青桔子自己动手在豆腐片上挤出汁液，其口感非一般豆腐，是爽爽的感觉好特别，值得一试！新品海蜇芥兰饺子包得满满鼓鼓的，但内馅主要还是芥兰丝，海蜇没丁点，不如直接点酸醋海蜇头更好吃又实在。'
# text = '|点心偏咸，代金券没有说明使用时间限制，确要下午三点才俾用。'
# text = '|#j交通#市区吃饭就是这样的，没有停车位，就门口几个，还有路边停的，抄不抄牌看运气了#环境#两层，装修就点都德的传统风格，很大，不用排队，第n次去的点都德。#虾饺#必点吧，里面很大很大的虾，朋友很爱，我还好，可能吃得太多了#百合蒸凤爪#每次一定都要点，最好吃就是这个，非常好吃??#蒸排骨#排骨有点甜，反而排骨下面的香芋非常好吃#三丝炒面#这个还不错，第一次点，炒得挺香的#三色小笼包#这个一般吧，反正里面又有大大的虾跟虾饺有点像那个馅#榴莲酥#这个很好吃，好香的榴莲味，榴莲肉满满。#油条#这个也是很好吃，就是分量好大，吃了一块好饱啊这次没有点红米卷，因为太多了，那个份量，每次都没吃完。星期六天价格会比平时贵一块钱，市区价格跟郊区价格也不一样。'

model_path = '/nas/lishengping/caiyun_projects/sentiment/output2/'
model = BertForMultipleChoice.from_pretrained(model_path, num_choices=4)
model.eval()
model.to(device)
tokenizer = tokenization.FullTokenizer(vocab_file=model_path + 'vocab.txt', do_lower_case=True)
max_seq_length = 200

predict_data = pd.read_excel('/nas/lishengping/jupyter/external_projects/temp/data/test.xlsx')
predict_results = []
for index, row in predict_data.iterrows():
    text = row['Content_review']
    passage_tokens = ["[CLS]"] + tokenizer.tokenize(text)[:max_seq_length - 2] + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(passage_tokens)
    input_mask = len(input_ids) * [1]
    segment_ids = [0] * len(input_ids)

    input_ids = torch.tensor(input_ids).view(1, -1).to(device)
    input_mask = torch.tensor(input_mask).view(1, -1).to(device)
    segment_ids = torch.tensor(segment_ids).view(1, -1).to(device)
    assert len(segment_ids) == len(input_ids) == len(input_mask)
    # print(f'input_ids: {input_ids}')
    # print(f'input_mask: {input_mask}')
    # print(f'segment_ids: {segment_ids}')
    logits = model(input_ids, segment_ids, input_mask)
    pred = torch.sigmoid(logits.cpu()).view(-1)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    label_strs = []
    for i, l in enumerate(pred.tolist()):
        if l:
            label_str = label_map[i]
            label_strs.append(label_str)
    label_strs = ' '.join(label_strs)
    print(f'原始文本: {text}')
    print(f'标签: {label_strs}\n\n')
    predict_results.append(label_strs)

predict_data['predict_label'] = predict_results
predict_data.to_excel('predict_results.xlsx', index=False)