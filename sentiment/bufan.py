"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
import re
import time
import copy
# from jieba import posseg

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader

import tokenization
from modeling import BertConfig, BertForSequenceClassification
from optimization import BertAdam
"""
====================================================================================
输入的数据格式：

__label__0  今天感冒了，怎么办？
__label__1  今天感冒了，怎么办？
__label__2  今天感冒了，怎么办？

====================================================================================
"""

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

WEIGHTS_NAME = 'pytorch_model.bin'
CONFIG_NAME = 'bert_config.json'
VOCAB_NAME = 'vocab.txt'


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, id, passage, label=None):
        self.id = id
        self.passage = passage
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class ParaDataloader(object):

    def __init__(self, examples: list):
        self.examples = examples
        self.example_size = len(examples)

    def __getitem__(self, item):
        return (torch.tensor(self.examples[item].input_ids, dtype=torch.long),
                torch.tensor(self.examples[item].input_mask, dtype=torch.long),
                torch.tensor(self.examples[item].segment_ids, dtype=torch.long),
                torch.tensor(self.examples[item].label_id, dtype=torch.long))

    def __len__(self):
        return self.example_size


class DataProcessor(object):
    """Processor for the CoLA data set (GLUE version)."""
    eval_dict = {}
    pos_vocab = {}

    def __init__(self):
        self.symbols = {
            '……': '...',
            "—": '-',
            '“': '"',
            '”': '"',
            '…': '...',
            '‘': '"',
            '’': '"',
            '展开全文c': ' '
        }

    def data_deal(self, passage):
        for k, v in self.symbols.items():
            passage = passage.replace(k, v)
        # p = re.sub(r'//@.*?:', ' ', passage)
        return passage

    #
    # def build_name(self):
    #     names = set()
    #     with open('../datas/names.txt', 'r', encoding='utf-8') as f:
    #         for name in f.readlines():
    #             name = name.strip('\n')
    #             names.add(name)
    #     print(f'names size {len(names)}')
    #     return list(names)

    def get_examples(self, path, data_type='train'):
        """Creates examples for the training and dev sets."""
        examples = []
        label = None
        with open(path, 'r', encoding='utf-8') as reader:
            for index, line in enumerate(reader):
                # try:
                line = line.strip('\n').strip().split('\t')
                if len(line) != 2:
                    continue
                passage = line[1]
                passage = self.data_deal(passage)
                if not passage or passage == 'nan':
                    continue
                # passage = re.sub(r'\?+|？+', '。', passage)
                # passage = re.sub(r'。+', '。', passage)
                label = int(line[0][-1])
                if label == 1:
                    continue
                if label == 2:
                    label -= 1
                # print(f'passage = {passage}')
                example = InputExample(id=index, passage=passage, label=label)
                examples.append(example)

                # example = InputExample(id=index, passage=passage, label=label)
                # examples.append(example)
                # except:
                #     logger.debug(f'label = {label}')
                #     logger.debug(f'line = {line}')

        random.shuffle(examples)
        logger.info(f'total {data_type} data {len(examples)}')
        return examples

    def get_eval_examples(self, path):
        """Creates examples for the training and dev sets."""
        examples = []
        label = None
        with open(path, 'r', encoding='utf-8') as reader:
            for index, line in enumerate(reader):
                # try:
                line = line.strip('\n').strip().split('\t')
                if len(line) != 2:
                    continue
                passage = line[1]
                passage = self.data_deal(passage)
                if not passage or passage == 'nan':
                    continue
                # passage = re.sub(r'\?+|？+', '。', passage)
                # passage = re.sub(r'。+', '。', passage)
                id = int(line[0][:-3])
                label = int(line[0][-1])
                if labels == 1:
                    continue
                if label == 2:
                    label -= 1
                self.eval_dict[id] = (passage, label)
                example = InputExample(id=id, passage=passage, label=id)
                examples.append(example)
                # except:
                #     logger.debug(f'label = {label}')
                #     logger.debug(f'line = {line}')
        logger.info(f'total eval data {len(examples)}')
        return examples

    def get_predict_examples(self, path):
        """Creates examples for the training and dev sets."""
        examples = []
        # datas = pd.read_excel(path, encoding='gbk')
        datas = pd.read_excel(path)

        count = 0
        for author, t,like_count, passage in zip(datas['author'], datas['time'],
                                                  datas['like_count'], datas['content']):
            passage = self.data_deal(passage)
            if not passage or isinstance(passage, float):
                continue
            # passage = re.sub(r'\?+|？+', '。', passage)
            # passage = re.sub(r'。+', '。', passage)
            self.eval_dict[count] = (passage, 100, author, like_count, t)
            example = InputExample(id=count, passage=passage, label=count)
            count += 1
            examples.append(example)
        logger.info(f'total predict data {len(examples)}')
        return examples


def convert_examples_to_features(args, examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    error = 0
    for (ex_index, example) in enumerate(examples):
        # try:
        passage_tokens = ["[CLS]"] + tokenizer.tokenize(
            example.passage)[:max_seq_length - 2] + ["[SEP]"]

        # except:
        #     error += 1
        #     logger.debug(f'error = {example.label}')
        #     continue

        input_ids = tokenizer.convert_tokens_to_ids(passage_tokens)

        segments_ids = [0] * len(input_ids)
        mask_ids = len(input_ids) * [1]
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            segments_ids.append(0)
            mask_ids.append(0)

        assert len(segments_ids) == len(input_ids) == len(mask_ids)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=mask_ids,
                          segment_ids=segments_ids,
                          label_id=example.label))
    # if args.do_train:
    #     num_batch = len(features) // args.train_batch_size
    #     features = features[:num_batch * args.train_batch_size]
    #     random.shuffle(features)
    # else:
    #     num_batch = len(features) // args.predict_batch_size
    #     features = features[:num_batch * args.predict_batch_size]

    logger.debug(f'input_ids = {features[-1].input_ids}')
    logger.debug(f'input_mask = {features[-1].input_mask}')
    logger.debug(f'segment_ids = {features[-1].segment_ids}')
    logger.debug(f'label_ids = {features[-1].label_id}')

    logger.info(f'error data = {error}')
    print(f'feature length = {len(features)}')
    return features


def save_checkpoint(model, epoch, output_dir):
    weights_name, ext = os.path.splitext(WEIGHTS_NAME)
    save_comment = f'{epoch:04d}'
    weights_name += f'-{save_comment}{ext}'
    output_model_file = os.path.join(output_dir, weights_name)
    logging.info(f"Saving fine-tuned model to: {output_model_file}")
    state_dict = model.state_dict()
    for t_name in state_dict:
        t_val = state_dict[t_name]
        state_dict[t_name] = t_val.to('cpu')
    torch.save(state_dict, output_model_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The train file path")
    parser.add_argument("--eval_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The dev file path")
    parser.add_argument("--predict_file",
                        default=None,
                        type=str,
                        required=False,
                        help="The predict file path")
    parser.add_argument("--predict_result_file",
                        default='datas/result.csv',
                        type=str,
                        required=False,
                        help="The predict result file path")
    parser.add_argument("--bert_model",
                        default=None,
                        type=str,
                        required=True,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                        "This specifies the model architecture.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument(
        "--do_lower_case",
        default=False,
        action='store_true',
        help="Whether to lower case the input text. True for uncased models, False for cased models."
    )
    parser.add_argument(
        "--max_seq_length",
        default=250,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--load_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to run load checkpoint.")
    parser.add_argument("--num_labels", default=1, type=int, help="mapping classify nums")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--epoches", default=6, type=int, help="Total epoch numbers for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=6.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                        "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.")

    args = parser.parse_args()
    vocab_path = os.path.join(args.bert_model, VOCAB_NAME)
    # bert_config = BertConfig.from_json_file(vocab_path)
    data_processor = DataProcessor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # if args.do_train:
    #     if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #         raise ValueError("Output directory ({}) already exists and is not empty.".format(
    #             args.output_dir))
    #     else:
    #         os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=args.do_lower_case)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=2)
    for k, v in model.state_dict().items():
        print(f'k = {k}, v.grad = {v.grad}')
    model.to(device)

    # model = torch.nn.DataParallel(model)

    def evaluating(model, eval_dataloader):
        model.eval()
        eval_loss = 0
        logits, labels = [], []
        for step, batch in enumerate(eval_dataloader):
            input_ids, input_mask, segment_ids, label_ids = [b.to(device) for b in batch]
            with torch.no_grad():
                loss, logit = model(input_ids, segment_ids, input_mask, label_ids)
                loss = loss.mean()
            eval_loss = loss * args.gradient_accumulation_steps if step == 0 else eval_loss + loss * args.gradient_accumulation_steps
            logit = torch.argmax(logit, dim=-1)
            logits.extend(logit.tolist())
            labels.extend(label_ids.tolist())
        return (eval_loss.item() / step, logits, labels)

    def predicting(model, dataloader):
        model.eval()
        logits, example_ids = [], []
        for step, batch in enumerate(dataloader):
            if step % 100 == 0:
                print(f'当前预测进度： {step}/{len(dataloader)}')
            input_ids, input_mask, segment_ids, label_ids = [b.to(device) for b in batch]
            with torch.no_grad():
                logit = model(input_ids, segment_ids, input_mask)
            logit = torch.argmax(logit, dim=-1)
            logits.extend(logit.tolist())
            example_ids.extend(label_ids.tolist())
        return logits, example_ids

    def eval_meric(model, data_loader):
        eval_loss, all_logits, all_labels = evaluating(model, data_loader)
        accuracy(all_labels, all_logits)
        logger.info(f'Average eval loss = {eval_loss}')
        return eval_loss

    def write_predict_file(model, data_loader, file_path):
        """
        写入预测文件： 格式：'五彩滨云-final.csv'
        """
        logits, ids = predicting(model, data_loader)
        assert len(ids) == len(logits)
        logger.info(
            f'zero nums {logits.count(0)}, one nums {logits.count(1)}, two nums {logits.count(2)}')
        labels = [data_processor.eval_dict[id][1] for id, logit in zip(ids, logits)]
        # if not args.do_eval:
        #     logits = [i - 1 for i in logits]
        #     data_df = pd.DataFrame({'id': ids, 'y': logits})
        # else:
        assert len(labels) == len(logits)
        # accuracy(labels, logits)
        passages = [data_processor.eval_dict[id][0] for id, logit in zip(ids, logits)]
        autors = [data_processor.eval_dict[id][2] for id, logit in zip(ids, logits)]
        like_counts = [data_processor.eval_dict[id][3] for id, logit in zip(ids, logits)]
        times = [data_processor.eval_dict[id][4] for id, logit in zip(ids, logits)]

        assert len(labels) == len(passages)
        match_array = np.array((logits)) == np.array(labels)
        match_list = match_array.tolist()
        data_df = pd.DataFrame({
            'id': ids,
            'pred': logits,
            'time': times,
            'match': '',
            'autors': autors,
            'like_counts': like_counts,
            'passage': passages
        })
        data_df.to_csv(file_path, index=None)

    eval_examples = data_processor.get_examples(args.eval_file, data_type='eval')

    eval_features = convert_examples_to_features(args, eval_examples, args.max_seq_length,
                                                 tokenizer)
    eval_loader = ParaDataloader(eval_features)
    eval_loader = DataLoader(eval_loader, shuffle=False, batch_size=args.eval_batch_size)

    if 1:
        # 数据读取
        train_examples = data_processor.get_examples(args.train_file, data_type='train')

        # 特征转换
        train_features = convert_examples_to_features(args, train_examples, args.max_seq_length,
                                                      tokenizer)

        num_train_steps = int(
            len(train_features) // args.train_batch_size // args.gradient_accumulation_steps *
            args.num_train_epochs)

        # 数据loader
        train_loader = ParaDataloader(train_features)

        # 数据并行loader输入格式
        train_loader = DataLoader(train_loader, shuffle=True, batch_size=args.train_batch_size)

        model.zero_grad()
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [{
            'params': [p for n, p in param_optimizer if n not in no_decay],
            'weight_decay_rate': 0.01
        }, {
            'params': [p for n, p in param_optimizer if n in no_decay],
            'weight_decay_rate': 0.0
        }]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_steps)
        tr_loss = None
        total_step = 0

        for epoch in range(args.epoches):
            model.train()
            min_eval_loss = 10000
            for step, batch in enumerate(train_loader):
                input_ids, input_mask, segment_ids, label_ids = [b.to(device) for b in batch]

                loss, _ = model(input_ids, segment_ids, input_mask, label_ids)
                loss = loss.mean()
                print(f'step: {step} loss = {loss}')
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss = loss * args.gradient_accumulation_steps if step == 0 else tr_loss + loss * args.gradient_accumulation_steps
                optimizer.step()
                optimizer.zero_grad()
                total_step += 1
                if total_step % 500 == 0 and total_step > 10:
                    eval_loss = eval_meric(model, eval_loader)
                    if eval_loss < min_eval_loss:
                        save_checkpoint(model, total_step, args.output_dir)

    if args.do_predict:
        if args.load_checkpoint:
            state_dict = torch.load('output/pytorch_model-0004.bin')
            model.load_state_dict(state_dict)
        logger.info(f'Start to predict......')
        if args.do_eval:
            predict_examples = data_processor.get_eval_examples(args.eval_file)
        else:
            predict_examples = data_processor.get_predict_examples(args.predict_file)

        predict_features = convert_examples_to_features(args, predict_examples, args.max_seq_length,
                                                        tokenizer)
        predict_loader = ParaDataloader(predict_features)
        predict_loader = DataLoader(predict_loader, shuffle=False, batch_size=args.eval_batch_size)
        write_predict_file(model, predict_loader, args.predict_result_file)


def accuracy(labels, logits):
    """
    计算f1值
    :param labels:
    :param logits:
    :return:
    """
    assert len(labels) == len(logits), "logits data size must be equal labels"

    labels_zero_nums = labels.count(0)
    labels_one_nums = labels.count(1)
    # labels_two_nums = labels.count(2)

    logits_zero_nums = logits.count(0)
    logits_one_nums = logits.count(1)
    # logits_two_nums = logits.count(2)

    acc_counter = defaultdict(int)

    for label, logit in zip(labels, logits):
        if label == logit:
            if label == 0:
                acc_counter['0'] += 1
            elif label == 1:
                acc_counter['1'] += 1
            # elif label == 2:
            #     acc_counter['2'] += 1

    zero_recall = acc_counter["0"] / labels_zero_nums if labels_zero_nums != 0 else 0
    zero_precision = acc_counter["0"] / logits_zero_nums if logits_zero_nums != 0 else 0
    zero_f1 = 2 * zero_recall * zero_precision / (zero_recall + zero_precision) if (
        zero_recall + zero_precision) != 0 else 0
    logger.info(f'\n')
    logger.info(
        f'zero ---- recall {round(zero_recall, 4)},  precision {round(zero_precision, 4)}, f1 {round(zero_f1, 4)}'
    )

    one_recall = acc_counter["1"] / labels_one_nums if labels_one_nums != 0 else 0
    one_precision = acc_counter["1"] / logits_one_nums if logits_one_nums != 0 else 0
    one_f1 = 2 * one_recall * one_precision / (one_recall + one_precision) if (
        one_recall + one_precision) != 0 else 0
    logger.info(
        f'one ---- recall {round(one_recall, 4)},  precision {round(one_precision, 4)}, f1 {round(one_f1, 4)}'
    )

    # two_recall = acc_counter["2"] / labels_two_nums if labels_two_nums != 0 else 0
    # two_precision = acc_counter["2"] / logits_two_nums if logits_two_nums != 0 else 0
    # two_f1 = 2 * two_recall * two_precision / (two_recall + two_precision) if (
    #     two_recall + two_precision) != 0 else 0
    # logger.info(
    #     f'two ---- recall {round(two_recall, 4)},  precision {round(two_precision, 4)}, f1 {round(two_f1, 4)}'
    # )

    logger.info(f'average f1 {round((zero_f1 + one_f1) / 2, 4)}')


if __name__ == "__main__":
    main()
