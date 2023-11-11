import logging
import os

import numpy as np
import pandas as pd
import torch

import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm


def preprocessing_raw_datasets(raw_dataset, task_name, tokenizer, max_seq_length):
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
        "ax": ("premise", "hypothesis"),
    }

    sentence1_key, sentence2_key = task_to_keys[task_name]  # 'sentence1' 'sentence2'

    def preprocess_function(examples):
        sentences = (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        inputs = tokenizer(*sentences, padding=True, max_length=max_seq_length, truncation=True)
        return inputs

    encoded_dataset = raw_dataset.map(preprocess_function, batched=True)
    encoded_dataset = encoded_dataset.remove_columns([sentence1_key, 'idx']) if sentence2_key is None else encoded_dataset.remove_columns([sentence1_key, sentence2_key, 'idx'])
    # encoded_dataset = encoded_dataset.rename_column('label', 'labels')

    return encoded_dataset


def val(node, val_dataset):
    node.model.to(node.device).eval()

    val_dataset = preprocessing_raw_datasets(val_dataset, node.args.task_name, node.tokenizer, node.max_seq_length)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=node.batch_size, collate_fn=node.data_collator)

    eval_loss = 0.
    metric = evaluate.load(node.args.dataset, node.args.task_name)
    for batch in tqdm(val_dataloader, desc='Evaluating'):
        batch = {k: v.to(node.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = node.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        eval_loss += loss
        if node.args.num_classes == 1:
            prediction = logits.squeeze()
        else:
            prediction = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=prediction, references=batch['labels'])

    eval_loss = eval_loss / len(val_dataloader)
    eval_results = metric.compute()

    node.model.cpu()
    return eval_loss, eval_results


def test(node, test_dataset):
    node.model.to(node.device).eval()

    test_dataset = preprocessing_raw_datasets(test_dataset, node.args.task_name, node.tokenizer, node.max_seq_length)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=node.batch_size, collate_fn=node.data_collator)

    predictions = None
    for batch in tqdm(test_dataloader, desc='Predicting'):
        del batch['labels']
        batch = {k: v.to(node.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = node.model(**batch)
        logits = outputs.logits

        if node.args.num_classes == 1:
            prediction = logits.squeeze()
        else:
            prediction = torch.argmax(logits, dim=-1)

        # for predicting
        if predictions is None:
            predictions = prediction.detach().cpu()
        else:
            predictions = torch.cat([predictions, prediction.detach().cpu()], dim=0)

    node.model.cpu()
    return predictions


class Recorder:
    def __init__(self, args):
        self.args = args

        self.val_acc = pd.DataFrame(columns=range(args.K + 1))
        self.current_acc = {k: None for k in range(args.K + 1)}
        self.best_acc = torch.zeros(self.args.K + 1)
        self.get_a_better = torch.zeros(self.args.K + 1)

    def evaluate(self, node, val_dataset, val_mismatched_dataset):
        if self.args.task_name != 'mnli':  # for other tasks
            val_loss, val_results = val(node, val_dataset)
            logging.info('val_loss={}, val_results={}'.format(val_loss, val_results))
        else:  # for mnli
            # for mnli-m
            val_loss, val_results = val(node, val_dataset)
            logging.info('matched-mnli: val_loss={}, val_results={}'.format(val_loss, val_results))
            # for mnli-mm
            val_mismatched_loss, val_mismatched_results = val(node, val_mismatched_dataset)
            logging.info('mismatched-mnli: val_loss={}, val_results={}'.format(val_mismatched_loss, val_mismatched_results))

        if self.args.task_name == 'cola':
            self.current_acc[node.id] = '{:.1f}'.format(val_results['matthews_correlation'] * 100)
        elif self.args.task_name == 'mnli':
            self.current_acc[node.id] = '{:.1f}/{:.1f}'.format(val_results['accuracy'] * 100, val_mismatched_results['accuracy'] * 100)
        elif self.args.task_name == 'mrpc' or node.args.task_name == 'qqp':
            self.current_acc[node.id] = '{:.1f}/{:.1f}'.format(val_results['f1'] * 100, val_results['accuracy'] * 100)
        elif self.args.task_name == 'stsb':
            self.current_acc[node.id] = '{:.1f}/{:.1f}'.format(val_results['pearson'] * 100, val_results['spearmanr'] * 100)
        else:  # for qnli, rte, sst2, wnli
            self.current_acc[node.id] = '{:.1f}'.format(val_results['accuracy'] * 100)

    def predict(self, node, test_dataset, test_mismatched_dataset, ax_dataset):
        output_eval_dir = os.path.join(self.args.submission_dir, '{}_{}'.format(node.name, node.model_type))
        os.makedirs(output_eval_dir, exist_ok=True)
        if self.args.task_name != 'mnli':  # for other tasks
            predictions = test(node, test_dataset)
            label_map = {i: label for i, label in enumerate(self.args.label_list)}
            if self.args.task_name == 'sst2':
                file_name = 'sst-2'.upper()
            elif self.args.task_name == 'stsb':
                file_name = 'sts-b'.upper()
            elif self.args.task_name == 'cola':
                file_name = 'CoLA'
            else:
                file_name = self.args.task_name.upper()
            output_eval_file = os.path.join(output_eval_dir, file_name + '.tsv')
            with open(output_eval_file, 'w') as f:
                f.write("index\tprediction\n")
                for index, prediction in enumerate(tqdm(predictions)):
                    prediction = prediction.item()
                    if self.args.task_name == 'stsb':
                        prediction = round(prediction, 3)
                    else:
                        prediction = label_map[prediction]
                    f.write('%s\t%s\n' % (index, str(prediction)))
                f.close()
            logging.info("Save the predictions to {}".format(output_eval_file))
        else:  # for mnli
            # for mnli-m
            predictions = test(node, test_dataset)
            label_map = {i: label for i, label in enumerate(self.args.label_list)}
            file_name = self.args.task_name.upper() + '-m'
            output_eval_file = os.path.join(output_eval_dir, file_name + '.tsv')
            with open(output_eval_file, 'w') as f:
                f.write("index\tprediction\n")
                for index, prediction in enumerate(tqdm(predictions)):
                    prediction = prediction.item()
                    if self.args.task_name == 'stsb':
                        prediction = round(prediction, 3)
                    else:
                        prediction = label_map[prediction]
                    f.write('%s\t%s\n' % (index, str(prediction)))
                f.close()
            logging.info("Save the predictions to {}".format(output_eval_file))
            # for mnli-mm
            predictions = test(node, test_mismatched_dataset)
            file_name = self.args.task_name.upper() + '-mm'
            output_eval_file = os.path.join(output_eval_dir, file_name + '.tsv')
            with open(output_eval_file, 'w') as f:
                f.write("index\tprediction\n")
                for index, prediction in enumerate(tqdm(predictions)):
                    prediction = prediction.item()
                    if self.args.task_name == 'stsb':
                        prediction = round(prediction, 3)
                    else:
                        prediction = label_map[prediction]
                    f.write('%s\t%s\n' % (index, str(prediction)))
                f.close()
            logging.info("Save the predictions to {}".format(output_eval_file))
            # for ax
            predictions = test(node, ax_dataset)
            file_name = 'AX'
            output_eval_file = os.path.join(output_eval_dir, file_name + '.tsv')
            with open(output_eval_file, 'w') as f:
                f.write("index\tprediction\n")
                for index, prediction in enumerate(tqdm(predictions)):
                    prediction = prediction.item()
                    if self.args.task_name == 'stsb':
                        prediction = round(prediction, 3)
                    else:
                        prediction = label_map[prediction]
                    f.write('%s\t%s\n' % (index, str(prediction)))
                f.close()
            logging.info("Save the predictions to {}".format(output_eval_file))

    def save_model(self, node):
        model_to_save = node.model.module if hasattr(node.model, 'module') else node.model
        file_name = os.path.join(self.args.model_dir, self.args.task_name, '{}_{}'.format(node.name, node.model_type))
        model_to_save.save_pretrained(file_name)
        node.tokenizer.save_pretrained(file_name)

    def save_record(self):
        self.val_acc.loc[len(self.val_acc)] = self.current_acc
        print(self.val_acc)
        self.val_acc.to_csv(os.path.join(self.args.record_dir, '{}.csv'.format(self.args.task_name)))
