import sys

import numpy as np
import torch

from datasets import load_dataset, concatenate_datasets
from transformers import set_seed


class DatasetPartition:
    def __init__(self, args):
        self.args = args

        if args.dataset == 'glue':
            raw_datasets = load_dataset(args.dataset, name=args.task_name, cache_dir=args.data_dir)

            # Datasets
            if args.task_name == 'mnli':
                train_dataset = raw_datasets['train']
                #
                self.val_dataset = raw_datasets['validation_matched']
                self.test_dataset = raw_datasets['test_matched']
                self.val_mismatched_dataset = raw_datasets['validation_mismatched']
                self.test_mismatched_dataset = raw_datasets['test_mismatched']
                self.ax_dataset = load_dataset(args.dataset, name='ax', cache_dir=args.data_dir)['test']
            else:
                train_dataset = raw_datasets['train']
                self.val_dataset = raw_datasets['validation']
                self.test_dataset = raw_datasets['test']

            # Labels
            if args.task_name == 'cola':
                args.label_list = ["0", "1"]
            elif args.task_name == 'mnli':  # or args.task_name == 'ax'
                args.label_list = ["entailment", "neutral", "contradiction"]
            elif args.task_name == 'mrpc':
                args.label_list = ["0", "1"]
            elif args.task_name == 'qnli':
                args.label_list = ["entailment", "not_entailment"]
            elif args.task_name == 'qqp':
                args.label_list = ["0", "1"]
            elif args.task_name == 'rte':
                args.label_list = ["entailment", "not_entailment"]
            elif args.task_name == 'sst2':
                args.label_list = ["0", "1"]
            elif args.task_name == 'stsb':
                args.label_list = [None]
            elif args.task_name == 'wnli':
                args.label_list = ["0", "1"]
            args.num_classes = len(args.label_list)

            # Partition `train_dataset` into `train_dataset` and `public_dataset`
            private_public_dataset = train_dataset.train_test_split(test_size=args.public_ratio)
            train_dataset = private_public_dataset['train']
            public_datasets = private_public_dataset['test']
            del private_public_dataset
            # Partition `public_dataset` into `labeled_public_dataset` and `unlabeled_public_dataset`
            labeled_unlabeled_public_dataset = public_datasets.train_test_split(test_size=args.labeled_public_ratio)
            self.public_dataset = labeled_unlabeled_public_dataset['train']
            self.query_dataset = labeled_unlabeled_public_dataset['test']
            del public_datasets, labeled_unlabeled_public_dataset

            if args.task_name == 'stsb':
                self.train_datasets = train_dataset
                self.train_datasets = []
                for k in range(args.K):
                    self.train_datasets.append(train_dataset.shard(num_shards=args.K, index=k))
            else:
                if args.iid:  # iid distribution
                    num_train = [int(len(train_dataset) / args.K) for _ in range(args.K)]
                    cumsum_train = torch.tensor(num_train).cumsum(dim=0).tolist()
                    idx_train = range(len(train_dataset['label']))
                    split_train_dataset = []
                    for off, l in zip(cumsum_train, num_train):
                        split_train_dataset.append(train_dataset.filter(lambda example, idx: idx in idx_train[off - l:off], with_indices=True))
                else:  # non-iid with Dirichlet distribution
                    train_labels = np.array(train_dataset['label'])
                    idx_batch_train = [[] for _ in range(args.K)]
                    for c in range(args.num_classes):
                        # get a list of batch indexes which are belong to label k:
                        idx_c = np.where(train_labels == c)[0]
                        np.random.shuffle(idx_c)
                        # using dirichlet distribution to determine the unbalanced proportion for each client (num_clients in total)
                        proportions = np.random.dirichlet(np.repeat(args.alpha, args.K))
                        # get the index in idx_k according to the dirichlet distribution
                        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
                        # generate the batch list for each client
                        idx_batch_train = [idx_k + idx.tolist() for idx_k, idx in zip(idx_batch_train, np.split(idx_c, proportions))]
                    client_dataidx_train = []
                    total = 0
                    for k in range(args.K):
                        np.random.shuffle(idx_batch_train[k])
                        client_dataidx_train.append(idx_batch_train[k])
                        total += len(idx_batch_train[k])
                    assert total == len(train_labels)

                    split_train_dataset = []
                    for dataidx in client_dataidx_train:
                        split_train_dataset.append(train_dataset.filter(lambda example, idx: idx in dataidx, with_indices=True))
                    self.train_datasets = split_train_dataset
                    del train_dataset, split_train_dataset
        else:
            sys.exit("Not implemented dataset, code exit, re-run to use correct dataset")


if __name__ == '__main__':
    class args:
        seed = 42
        dataset = 'glue'
        data_dir = 'data'
        task_name = 'stsb'
        iid = False
        alpha = 1
        K = 10
        public_ratio = 0.5
        labeled_public_ratio = 0.1


    set_seed(args.seed)

    glue = DatasetPartition(args)
    train_datasets = glue.train_datasets
    public_dataset = glue.public_dataset
    query_dataset = glue.query_dataset
    val_dataset = glue.val_dataset
    test_dataset = glue.test_dataset
    #
    val_mismatched_dataset = glue.val_mismatched_dataset if args.task_name == 'mnli' else None  # except mnli: None
    test_mismatched_dataset = glue.test_mismatched_dataset if args.task_name == 'mnli' else None  # except mnli: None
    ax_dataset = glue.ax_dataset if args.task_name == 'mnli' else None  # except mnli: None

    if train_datasets is not None:
        print('length of train_datasets: {}'.format([len(train_datasets[n]) for n in range(args.K)]))
    if public_dataset is not None:
        print('length of public_dataset: {}'.format(len(public_dataset)))
    if query_dataset is not None:
        print('length of query_dataset: {}'.format(len(query_dataset)))
    if val_dataset is not None:
        print('length of val_dataset: {}'.format(len(val_dataset)))
        if val_mismatched_dataset is not None:
            print('length of val_mismatched_dataset: {}'.format(len(val_mismatched_dataset)))
    if test_dataset is not None:
        print('length of test_dataset: {}'.format(len(test_dataset)))
        if test_mismatched_dataset is not None:
            print('length of test_mismatched_dataset: {}'.format(len(test_mismatched_dataset)))
        if ax_dataset is not None:
            print('length of ax_dataset: {}'.format(len(ax_dataset)))
