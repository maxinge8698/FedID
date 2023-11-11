import copy
import logging
import sys

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from datasets import concatenate_datasets

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler

from tqdm import tqdm


def init_model(name, model_type, num_classes):
    # config = AutoConfig.from_pretrained(model_type, num_labels=num_classes, finetuning_task=task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    total_params = total_params / 1000000
    logging.info('model parameters of %s_%s: %2.1fM' % (name, model_type, total_params))
    return tokenizer, model


def init_optimizer(optimizer_type, model, lr, weight_decay=0., momentum=0.9):
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(momentum, 0.999), eps=1e-8)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(momentum, 0.999), eps=1e-8)
    else:
        sys.exit("Not implemented optimizer, code exit, re-run to use correct optimizer")
    return optimizer


def init_scheduler(scheduler_type, optimizer, num_warmup_steps=None, num_training_steps=None):
    if scheduler_type == 'linear':
        scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    elif scheduler_type == "cosine":  # cosine
        scheduler = get_scheduler('cosine', optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    else:
        sys.exit("Not implemented learning rate scheduler, code exit, re-run to use correct scheduler")
    return scheduler


def preprocessing_raw_datasets(raw_dataset, task_name, tokenizer, max_seq_length, logits=None):
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

    if logits is not None:
        if type(logits) == list:
            for k in range(len(logits)):
                encoded_dataset = encoded_dataset.add_column('logits{}'.format(k), logits[k].tolist())
        else:
            encoded_dataset = encoded_dataset.add_column('logits', logits.tolist())

    return encoded_dataset


class Client:
    def __init__(self, args, id, model_type, train_dataset=None):
        self.args = args
        self.id = id
        self.name = 'client' + str(id)
        self.model_type = model_type

        self.device = args.device
        self.optimizer_type = args.optimizer
        self.scheduler_type = args.scheduler

        self.max_seq_length = args.max_seq_length
        self.batch_size = args.batch_size

        self.tokenizer, self.model = init_model(self.name, self.model_type, self.args.num_classes)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.train_dataset = train_dataset

        self.E = args.E
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
        self.warmup_steps = args.warmup_steps

    def fork(self, w_glob):
        self.model.load_state_dict(w_glob)

    def local_update(self):
        self.model.to(self.device).train()

        train_dataset = preprocessing_raw_datasets(self.train_dataset, self.args.task_name, self.tokenizer, self.max_seq_length)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=len(train_dataloader) * self.E)

        for epoch in range(self.E):
            train_loss = 0.
            metric = evaluate.load(self.args.dataset, self.args.task_name)
            for batch in tqdm(train_dataloader, desc='Iteration'):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                train_loss += loss.item()
                if self.args.num_classes == 1:
                    prediction = logits.squeeze()
                else:
                    prediction = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=prediction, references=batch['labels'])

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss = train_loss / len(train_dataloader)
            train_results = metric.compute()
            logging.info('Epoch {}/{}: train_loss={}, train_results={}'.format(epoch + 1, self.E, train_loss, train_results))

        self.model.cpu()

    def local_distillation(self, public_dataset, logits_glob):
        self.model.to(self.device).train()

        public_dataset = preprocessing_raw_datasets(public_dataset, self.args.task_name, self.tokenizer, self.max_seq_length, logits_glob)
        public_dataloader = DataLoader(public_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=len(public_dataloader) * self.E)

        for epoch in range(self.args.dis_epochs):
            train_loss = 0.
            metric = evaluate.load(self.args.dataset, self.args.task_name)
            for batch in tqdm(public_dataloader, desc='Distilling'):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                hard_label = batch.pop('labels')

                soft_label = batch.pop('logits')
                outputs = self.model(**batch)
                logits = outputs.logits
                loss = F.cross_entropy(logits, torch.argmax(soft_label, dim=-1))
                train_loss += loss.item()

                if self.args.num_classes == 1:
                    prediction = logits.squeeze()
                else:
                    prediction = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=prediction, references=hard_label)

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss = train_loss / len(public_dataloader)
            train_results = metric.compute()
            logging.info('Epoch {}/{}: train_loss={}, train_results={}'.format(epoch + 1, self.args.dis_epochs, train_loss, train_results))

        self.model.cpu()

    def compute_logits(self, public_dataset):
        self.model.to(self.device).eval()

        public_dataset = preprocessing_raw_datasets(public_dataset, self.args.task_name, self.tokenizer, self.max_seq_length)
        public_dataloader = DataLoader(public_dataset, shuffle=False, batch_size=self.batch_size, collate_fn=self.data_collator)
        logits = None
        for batch in tqdm(public_dataloader, desc='Predicting'):
            del batch['labels']
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logit = outputs.logits

            if logits is None:
                logits = logit.detach().cpu()
            else:
                logits = torch.cat([logits, logit.detach().cpu()], dim=0)

        self.model.cpu()
        return logits

    def compute_logits_batch(self, batch_dataset):
        self.model.to(self.device).train()

        batch_dataset = preprocessing_raw_datasets(batch_dataset, self.args.task_name, self.tokenizer, self.max_seq_length)
        batch_dataloader = DataLoader(batch_dataset, shuffle=False, batch_size=len(batch_dataset), collate_fn=self.data_collator)
        logits = None
        for batch in batch_dataloader:
            del batch['labels']
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # with torch.no_grad():
            outputs = self.model(**batch)
            logit = outputs.logits

            if logits is None:
                logits = logit.detach().cpu()
            else:
                logits = torch.cat([logits, logit.detach().cpu()], dim=0)

        self.model.cpu()
        return logits

    def query_update(self, dot_product, batch_dataset, batch_logits_local, weight, batch_logits_glob):
        self.model.to(self.device).train()

        batch_dataset = preprocessing_raw_datasets(batch_dataset, self.args.task_name, self.tokenizer, self.max_seq_length)
        batch_dataloader = DataLoader(batch_dataset, shuffle=False, batch_size=len(batch_dataset), collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.args.dis_lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=len(batch_dataloader) * self.args.dis_epochs)

        train_loss = 0.
        # metric = evaluate.load(self.args.dataset, self.args.task_name)
        for batch in batch_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            hard_labels = batch.pop('labels')

            outputs = self.model(**batch)
            logits = outputs.logits

            loss1 = weight * dot_product * F.cross_entropy(logits, torch.argmax(batch_logits_local, dim=-1).to(self.device))
            loss2 = F.cross_entropy(logits, torch.argmax(batch_logits_glob, dim=-1).to(self.device))
            loss = loss1 + loss2
            train_loss += loss.item()

            # if self.args.num_classes == 1:
            #     prediction = logits.squeeze()
            # else:
            #     prediction = torch.argmax(logits, dim=-1)
            # metric.add_batch(predictions=prediction, references=hard_labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss / len(batch_dataloader)
        # train_results = metric.compute()
        # logging.info('Epoch {}/{}: train_loss={}, train_results={}'.format(epoch + 1, self.args.dis_epochs, train_loss, train_results))
        # logging.info('train_loss={}, train_results={}'.format(train_loss, train_results))

        self.model.cpu()


class Server:
    def __init__(self, args, id, model_type, public_dataset=None):
        self.args = args
        self.id = id
        self.name = 'server'
        self.model_type = model_type

        self.device = args.device
        self.optimizer_type = args.optimizer
        self.scheduler_type = args.scheduler

        self.max_seq_length = args.max_seq_length
        self.batch_size = args.batch_size
        # self.loss_fn = nn.CrossEntropyLoss() if args.task_name != 'stsb' else nn.MSELoss()

        self.tokenizer, self.model = init_model(self.name, self.model_type, self.args.num_classes)  # 全局模型
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.public_dataset = public_dataset

    def merge(self, w_locals, weights):
        w_avg = copy.deepcopy(w_locals[0])
        for n in w_avg.keys():
            w_avg[n] = 0.
            for k in range(len(w_locals)):
                w_avg[n] += weights[k] * w_locals[k][n]
        return w_avg

    def centralized_training(self, train_dataset):
        self.model.to(self.device).train()

        train_dataset = preprocessing_raw_datasets(train_dataset, self.args.task_name, self.tokenizer, self.max_seq_length)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.args.lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=len(train_dataloader) * self.args.E)

        for epoch in range(self.args.E):
            train_loss = 0.
            metric = evaluate.load(self.args.dataset, self.args.task_name)
            for batch in tqdm(train_dataloader, desc='Iteration'):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                train_loss += loss.item()
                if self.args.num_classes == 1:
                    prediction = logits.squeeze()
                else:
                    prediction = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=prediction, references=batch['labels'])

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss = train_loss / len(train_dataloader)
            train_results = metric.compute()
            logging.info('Epoch {}/{}: train_loss={}, train_results={}'.format(epoch + 1, self.args.E, train_loss, train_results))

        self.model.cpu()

    def logit_ensemble(self, logits_locals, weights):
        logits_glob = torch.zeros_like(logits_locals[0])
        for k in range(len(logits_locals)):
            logits_glob += weights[k] * logits_locals[k]
        # logits_glob /= len(logits_locals)
        return logits_glob

    def logit_ensemble_with_ERA(self, logits_locals, weights):
        logits_glob = torch.zeros_like(logits_locals[0])
        for k in range(len(logits_locals)):
            logits_glob += weights[k] * logits_locals[k]
        # logits_glob /= len(logits_locals)
        T = 0.1
        logits_glob = torch.softmax(logits_glob / T, dim=-1)
        return logits_glob

    def batch_logit_ensemble(self, batch_logits_locals, weights):
        batch_logits_glob = torch.zeros_like(batch_logits_locals[0])
        for k in range(len(batch_logits_locals)):
            batch_logits_glob += weights[k] * batch_logits_locals[k]
        # batch_logits_glob /= len(batch_logits_locals)
        return batch_logits_glob

    def ensemble_distillation(self, public_dataset, logits_glob):
        self.model.to(self.device).train()

        public_dataset = preprocessing_raw_datasets(public_dataset, self.args.task_name, self.tokenizer, self.max_seq_length, logits_glob)
        public_dataloader = DataLoader(public_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.args.dis_lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=len(public_dataloader) * self.args.dis_epochs)

        for epoch in range(self.args.dis_epochs):
            train_loss = 0.
            metric = evaluate.load(self.args.dataset, self.args.task_name)
            for batch in tqdm(public_dataloader, desc='Distilling'):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                hard_label = batch.pop('labels')

                soft_label = batch.pop('logits')
                outputs = self.model(**batch)
                logits = outputs.logits
                if self.args.algorithm in ['fed_df', 'fed_ed']:
                    T = 1
                    loss = F.kl_div(F.log_softmax(logits / T, dim=-1), F.softmax(soft_label / T, dim=-1), reduction='batchmean') * (T ** 2)
                elif self.args.algorithm == 'fed_kd':
                    loss = F.mse_loss(logits, soft_label)
                elif self.args.algorithm == 'ds_fl':
                    loss = F.cross_entropy(logits, torch.argmax(soft_label, dim=-1))
                else:
                    sys.exit("Not implemented algorithm, code exit, re-run to use correct algorithm")

                train_loss += loss.item()

                if self.args.num_classes == 1:
                    prediction = logits.squeeze()
                else:
                    prediction = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=prediction, references=hard_label)

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss = train_loss / len(public_dataloader)
            train_results = metric.compute()
            logging.info('Epoch {}/{}: train_loss={}, train_results={}'.format(epoch + 1, self.args.dis_epochs, train_loss, train_results))
        self.model.cpu()

    def mhat_distillation(self, public_dataset, logits_locals, weights):
        self.model.to(self.device).train()

        public_dataset = preprocessing_raw_datasets(public_dataset, self.args.task_name, self.tokenizer, self.max_seq_length, logits_locals)
        public_dataloader = DataLoader(public_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.args.dis_lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=len(public_dataloader) * self.args.dis_epochs)

        for epoch in range(self.args.dis_epochs):
            train_loss = 0.
            metric = evaluate.load(self.args.dataset, self.args.task_name)
            for batch in tqdm(public_dataloader, desc='Distilling'):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                hard_label = batch.pop('labels')

                soft_label = []
                for k in range(len(logits_locals)):
                    soft_label.append(batch.pop('logits{}'.format(k)))

                outputs = self.model(**batch)
                logits = outputs.logits

                loss = 0.
                for k in range(len(logits_locals)):
                    tmp_kd_loss = weights[k] * F.cross_entropy(logits, torch.argmax(soft_label[k], dim=-1), reduction='mean')
                    loss += tmp_kd_loss
                train_loss += loss.item()

                if self.args.num_classes == 1:
                    prediction = logits.squeeze()
                else:
                    prediction = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=prediction, references=hard_label)

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss = train_loss / len(public_dataloader)
            train_results = metric.compute()
            logging.info('Epoch {}/{}: train_loss={}, train_results={}'.format(epoch + 1, self.args.dis_epochs, train_loss, train_results))

        self.model.cpu()

    def compute_logits(self, public_dataset):
        self.model.to(self.device).eval()

        public_dataset = preprocessing_raw_datasets(public_dataset, self.args.task_name, self.tokenizer, self.max_seq_length)
        public_dataloader = DataLoader(public_dataset, shuffle=False, batch_size=self.batch_size, collate_fn=self.data_collator)
        logits = None
        for batch in tqdm(public_dataloader, desc='Predicting'):
            del batch['labels']
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logit = outputs.logits

            if logits is None:
                logits = logit.detach().cpu()
            else:
                logits = torch.cat([logits, logit.detach().cpu()], dim=0)

        self.model.cpu()
        return logits

    def get_query_dataloader(self, query_dataset):
        query_dataset = preprocessing_raw_datasets(query_dataset, self.args.task_name, self.tokenizer, self.max_seq_length)
        query_dataloader = DataLoader(query_dataset, shuffle=True, batch_size=self.batch_size, collate_fn=self.data_collator)
        return query_dataloader

    def compute_query_loss(self, query_batch):
        self.model.to(self.device).train()

        batch = {k: v.to(self.device) for k, v in query_batch.items()}

        hard_label = batch.pop('labels')

        outputs = self.model(**batch)
        logits = outputs.logits
        loss = F.cross_entropy(logits.detach(), hard_label, reduction='mean')

        self.model.cpu()
        return loss

    def batch_distillation(self, batch_dataset, batch_logits_locals, weights):
        self.model.to(self.device).train()

        batch_dataset = preprocessing_raw_datasets(batch_dataset, self.args.task_name, self.tokenizer, self.max_seq_length, batch_logits_locals)
        batch_dataloader = DataLoader(batch_dataset, shuffle=True, batch_size=len(batch_dataset), collate_fn=self.data_collator)

        optimizer = init_optimizer(self.optimizer_type, self.model, self.args.dis_lr, weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        scheduler = init_scheduler(self.scheduler_type, optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=len(batch_dataloader) * self.args.dis_epochs)

        # for epoch in range(self.args.dis_epochs):
        train_loss = 0.
        # metric = evaluate.load(self.args.dataset, self.args.task_name)
        for batch in batch_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            hard_label = batch.pop('labels')
            soft_label = []
            for k in range(len(batch_logits_locals)):
                soft_label.append(batch.pop('logits{}'.format(k)))

            outputs = self.model(**batch)
            logits = outputs.logits

            loss = 0.
            for k in range(len(batch_logits_locals)):
                tmp_loss = weights[k] * F.cross_entropy(logits, torch.argmax(soft_label[k], dim=-1), reduction='mean')
                # tmp_loss = F.cross_entropy(logits, torch.argmax(soft_labels[k], dim=-1), reduction='mean')
                loss += tmp_loss

            train_loss += loss.item()

            # if self.args.num_classes == 1:
            #     prediction = logits.squeeze()
            # else:
            #     prediction = torch.argmax(logits, dim=-1)
            # metric.add_batch(predictions=prediction, references=hard_label)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_loss = train_loss / len(batch_dataloader)
        # train_results = metric.compute()
        # logging.info('Epoch {}/{}: train_loss={}, train_results={}'.format(epoch + 1, self.args.dis_epochs, train_loss, train_results))

        self.model.cpu()
        # return train_loss


if __name__ == '__main__':
    class args:
        num_classes = 2
        model_type = 'bert-base-uncased'
