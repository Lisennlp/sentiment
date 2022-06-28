"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from modeling import BertForSequenceClassification
from optimization import BertAdam
from sklearn.metrics import accuracy_score, classification_report


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

    def __init__(self):
        self.label_map = {'neg': 0, 'neu': 1, 'pos': 2}
        self.stop_words = open('/nas/lishengping/jupyter/en_stop_words.txt', 'r').readlines()
        self.stop_words = [w.strip() for w in  self.stop_words if w.strip()]

    def clean(self, text):
        text = text.replace('&#xD', '')
        text = text.replace('&#xA', '')
        text = text.replace('  ', ' ')
        text = text.replace('，', ',')
        # text = text.replace('|', '')
        return  text

    def get_examples(self, path, shuffle=True):
        """Creates examples for the training and dev sets."""
        examples = []
        label_counter = defaultdict(int)
        # path = '/nas/lishengping/temp/selected_train.csv'
        data = pd.read_excel(path)
        count = 0
        for index, row in data.iterrows():
            count += 1
            if not isinstance(row['text'], str):
                continue
            text = row['text'].strip()
            label = int(row['label']) 
            text = self.clean(text)
            # text_list = text.split()
            # text_list = [word for word in text_list if word not in self.stop_words]
            # text = ' '.join(text_list)
            if label not in list(range(5)):
                print(f'error data label = {label}')
                continue
            label_counter[label] += 1
            example = InputExample(id=index, passage=text, label=label)
            examples.append(example)
        if shuffle:
            random.shuffle(examples)
        # dev_data = examples[:len(examples) // 8]
        # train_data = examples[4 * len(examples) // 8:]
        print(f'label_counter = {label_counter}')
        return examples

    def get_predict_examples(self, path):
        examples = []
        data = pd.read_csv(path)
        for index, row in data.iterrows():
            text = row['Text'].strip()
            text = self.clean(text)
            example = InputExample(id=index, passage=text, label=0)
            examples.append(example)
        return examples

    def get_image_feature(self, path):
        import pickle
        data = pickle.load(open(path, 'rb'))
        return data

import pickle


def convert_examples_to_features(examples, max_seq_length, tokenizer, type_='train'):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    eval_counter = defaultdict(int)
    for (ex_index, example) in enumerate(examples):
        eval_counter[example.label] += 1
        if ex_index % 100 == 0:
            print(f'当前进度：{ex_index}/{len(examples)}')
        passage_tokens = ["[CLS]"] + tokenizer.tokenize(
            example.passage)[:max_seq_length - 2] + ["[SEP]"]
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

    logger.debug(f'input_ids = {features[-1].input_ids}')
    logger.debug(f'input_mask = {features[-1].input_mask}')
    logger.debug(f'segment_ids = {features[-1].segment_ids}')
    logger.debug(f'label_ids = {features[-1].label_id}\n\n')
    print(f'feature length = {len(features)}')
    print(f'eval_counter = {eval_counter} type = {type_}')
    pickle.dump(features, open(f'/nas/lishengping/temp/2_{type_}_features_cache_half.pkl', 'wb'))
    return features


def save_checkpoint(model, epoch, step, output_dir):
    weights_name, ext = os.path.splitext(WEIGHTS_NAME)
    save_comment = f'{epoch:02d}-{step}'
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
                        required=False,
                        help="The dev file path")
    parser.add_argument("--predict_file",
                        default=None,
                        type=str,
                        required=False,
                        help="The predict file path")
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
                        default=200,
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
    data_processor = DataProcessor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cuda:1'
    print(f'device: {device}')

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    args.num_train_epochs = int(args.num_train_epochs)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    def evaluating(model, eval_dataloader):
        model.eval()
        eval_loss = 0
        logits, labels = [], []
        for step, batch in enumerate(eval_dataloader):
            if step % 100 == 0:
                print(f'当前进度：{step}/{len(eval_dataloader)}')
            input_ids = batch['text_bert_indices'].to(device)
            segment_ids = batch['bert_segments_ids'].to(device)
            photo = batch['photo'].to(device)
            label_ids = batch['level'].to(device)
            with torch.no_grad():
                loss, logit = model(input_ids, segment_ids, labels=label_ids)
                loss = loss.mean()
            eval_loss = loss * args.gradient_accumulation_steps if step == 0 else eval_loss + loss * args.gradient_accumulation_steps
            logit = torch.argmax(logit, dim=-1)
            # right = (logit.view(-1) == label_ids.view(-1)).sum()
            logits.extend(logit.cpu().tolist())
            labels.extend(label_ids.cpu().tolist())
        model.train()
        return (eval_loss.item() / step, logits, labels)

    def eval_meric(model, data_loader):
        eval_loss, all_logits, all_labels = evaluating(model, data_loader)
        acc = accuracy_score(all_labels, all_logits)
        result = classification_report(all_labels, all_logits, target_names=[str(i) for i in range(5)], digits=4)
        # accuracy(all_labels, all_logits)
        logger.info(f'Average eval loss = {eval_loss}, acc = {acc}')
        print(result)
        return eval_loss

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=args.do_lower_case)

   
    train_features = data_processor.get_image_feature(args.train_file)
    eval_features = data_processor.get_image_feature(args.eval_file)

    eval_loader = DataLoader(eval_features, shuffle=False, batch_size=args.eval_batch_size)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels=5)

    for k, v in model.state_dict().items():
        print(f'k = {k}, v.grad = {v.grad}')
    model.to(device)
    # model = torch.nn.DataParallel(model)

    if args.do_train:
        train_loader = DataLoader(train_features, shuffle=True, batch_size=args.train_batch_size)
        num_train_steps = int(
            len(train_features) // args.train_batch_size // args.gradient_accumulation_steps *
            args.num_train_epochs)
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
        min_eval_loss = 10000
        for epoch in range(int(args.num_train_epochs)):
            total = 0
            total_right = 0

            model.train()
            for step, batch in enumerate(train_loader):
                input_ids = batch['text_bert_indices'].to(device)
                segment_ids = batch['bert_segments_ids'].to(device)
                photo = batch['photo'].to(device)
                label_ids = batch['level'].to(device)
                print(f'input_ids: {input_ids.shape}')

                loss, logits = model(input_ids, segment_ids, labels=label_ids)
                preds = logits.argmax(-1)
                loss = loss.mean()
                right = (preds == label_ids.view(-1)).sum()
                total_right += right.item()
                total = (step + 1) * args.train_batch_size
                # print(f'right = {right} total = {total} loss = {loss} label_ids = {label_ids}')
                acc = total_right / total
                # print(f'train acc = {acc} right = {right}')
                if step % 20 == 0:
                    print(f'epoch {epoch} step {step}/{len(train_loader)} loss = {loss} train acc = {acc}')
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss = loss * args.gradient_accumulation_steps if step == 0 else tr_loss + loss * args.gradient_accumulation_steps
                optimizer.step()
                optimizer.zero_grad()
            eval_loss = eval_meric(model, eval_loader)
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                # save_checkpoint(model, epoch, 0, args.output_dir)
                model.train()

    elif args.do_predict:
        new_data = pd.DataFrame()
        cache_test_file = '/nas/lishengping/temp/16_predict_features_cache_200.pkl'
        if os.path.exists(cache_test_file):
            print(F'加载缓存测试文件......')
            predcit_features = pickle.load(open(cache_test_file, 'rb'))
        else:
            predict_examples = data_processor.get_predict_examples(args.predict_file)
            predcit_features = convert_examples_to_features(predict_examples, args.max_seq_length, tokenizer, 'predict')
        predict_loader = ParaDataloader(predcit_features)
        predict_loader = DataLoader(predict_loader, shuffle=False, batch_size=args.eval_batch_size)
        data = pd.read_csv(args.predict_file)
        _, logits, _ = evaluating(model, predict_loader)
        print(f'length = {len(logits)}')
        data['Outcome'] = logits
        data.to_csv('output/result_2_00_16999_back.csv')
        new_data['Outcome'] = logits
        new_data['Id'] = data['Id']
        new_data.to_csv('output/result_2_00_16999.csv')


if __name__ == "__main__":
    main()
