import argparse
import logging
import math
import sys

import torch
import os
import numpy as np
from datasets import concatenate_datasets
from tqdm import tqdm

from dataset import DatasetPartition
from node import Server, Client
from recorder import Recorder

from transformers import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Common Hyper-parameters
    # Total
    parser.add_argument('--algorithm', type=str, default='fed_avg', help='Type of algorithms:{centralized, fed_avg, fed_df, fed_ed, fed_kd, mhat, ds_fl, fed_id}')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--K', type=int, default=10, help='Number of clients')
    parser.add_argument('--C', type=float, default=1, help='Fraction of clients')
    parser.add_argument('--R', type=int, default=10, help='Number of rounds')
    # Data
    parser.add_argument('--dataset', type=str, default='glue', help='Type of dataset: {glue}')
    parser.add_argument('--task_name', type=str, default='rte', help='Type of task: {cola, mnli, mrpc, qnli, qqp, rte, sst2, wnli}')
    parser.add_argument('--data_dir', type=str, default='data', help='Path of data dir')
    parser.add_argument('--public_ratio', type=float, default=0.5, help='Ratio of public dataset')
    parser.add_argument('--labeled_public_ratio', type=float, default=0.1, help='Ratio of labeled public dataset')
    parser.add_argument('--iid', action='store_true', default=False, help='iid data distribution or non-iid data with Dirichlet distribution')
    parser.add_argument('--alpha', type=float, default=1, help='radio of Dirichlet distribution')
    # Model
    parser.add_argument('--central_model', type=str, default='bert-base-uncased', help='Type of global model: {bert-base-uncased, bert-large-uncased, roberta-base, roberta-large}')
    # Output
    parser.add_argument("--output_dir", type=str, default="./saves/", help="The output directory where checkpoints/results/logs will be written.")
    parser.add_argument('--do_test', action='store_true', default=False, help='Whether to make predictions at the end of the last round.')

    # Specific Hyper-parameters
    # Data
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    # Model
    parser.add_argument('--local_models', type=str, default='bert-base-uncased,bert-base-uncased,bert-base-uncased,bert-base-uncased,bert-base-uncased,bert-base-uncased,bert-base-uncased,bert-base-uncased,bert-base-uncased,bert-base-uncased',
                        help='Type of local model: {distilbert-base-uncased, bert-base-uncased, bert-large-uncased, distilroberta-base, roberta-base, roberta-large}')
    # Optima
    parser.add_argument('--E', type=int, default=3, help='Number of local epochs')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Type of optimizer: {sgd, adam, adamw}')
    parser.add_argument('--scheduler', type=str, default='linear', help='Type of scheduler: {liner, cosine}')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate of local training')
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay for optimizer")
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum for optimizer')
    parser.add_argument("--warmup_steps", type=int, default=0, help="Step of training to perform learning rate warmup for if set for cosine and linear decay")

    #
    parser.add_argument('--dis_epochs', type=int, default=3, help='Number of distillation epochs')
    parser.add_argument('--dis_lr', type=float, default=2e-5, help='Learning rate of distillation ')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set dir
    args.model_dir = os.path.join(args.output_dir, args.algorithm, 'model')
    args.record_dir = os.path.join(args.output_dir, args.algorithm, 'record')
    args.log_dir = os.path.join(args.output_dir, args.algorithm, 'log')
    args.submission_dir = os.path.join(args.output_dir, args.algorithm, 'submission')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.record_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.submission_dir, exist_ok=True)

    # Set log
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[%(levelname)s](%(asctime)s) %(message)s",
                        level=logging.INFO,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[logging.FileHandler(os.path.join(args.log_dir, '{}.txt'.format(args.task_name))), logging.StreamHandler(sys.stdout)])

    # Set data
    glue = DatasetPartition(args)
    # private data
    train_datasets = glue.train_datasets
    # public data
    public_dataset = glue.public_dataset
    # query data for FedID
    query_dataset = glue.query_dataset
    # validation and test data
    val_dataset = glue.val_dataset
    test_dataset = glue.test_dataset
    #
    val_mismatched_dataset = glue.val_mismatched_dataset if args.task_name == 'mnli' else None  # except mnli: None
    test_mismatched_dataset = glue.test_mismatched_dataset if args.task_name == 'mnli' else None  # except mnli: None
    ax_dataset = glue.ax_dataset if args.task_name == 'mnli' else None  # except mnli: None

    # Set recoder
    recorder = Recorder(args)

    # Federated training
    logger.info('Running on %s', args.device)
    logger.info("algorithm: {}".format(args.algorithm))
    logger.info("dataset: {},\ttask: {},\tpublic_ratio: {},\tlabeled_public_ratio: {}".format(args.dataset, args.task_name, args.public_ratio, args.labeled_public_ratio))
    if args.iid:
        logger.info("iid: {}".format(args.iid))
    else:
        logger.info('iid: {},\talpha: {}'.format(args.iid, args.alpha))
    if train_datasets is not None:
        logger.info('length of train_datasets: {}'.format([len(train_datasets[k]) for k in range(args.K)]))
    if public_dataset is not None:
        logger.info('length of public_dataset: {}'.format(len(public_dataset)))
    if query_dataset is not None:
        logger.info('length of query_dataset: {}'.format(len(query_dataset)))
    if val_dataset is not None:
        logger.info('length of val_dataset: {}'.format(len(val_dataset)))
        if val_mismatched_dataset is not None:
            logger.info('length of val_mismatched_dataset: {}'.format(len(val_mismatched_dataset)))
    if test_dataset is not None:
        logger.info('length of test_dataset: {}'.format(len(test_dataset)))
        if test_mismatched_dataset is not None:
            logger.info('length of test_mismatched_dataset: {}'.format(len(test_mismatched_dataset)))
        if ax_dataset is not None:
            logger.info('length of ax_dataset: {}'.format(len(ax_dataset)))
    logger.info("num_clients: {},\tfraction: {}".format(args.K, args.C))
    logger.info("global_rounds: {}".format(args.R))
    logger.info("global_model: {}".format(args.central_model))
    args.local_models = args.local_models.split(',')
    logger.info("local_models: [{}]".format(', '.join(args.local_models)))
    logger.info("batch_size: {},\tmax_seq_length: {},\tlocal_epochs: {},\tlr: {}".format(args.batch_size, args.max_seq_length, args.E, args.lr))
    if args.algorithm in ['fed_df', 'fed_ed', 'fed_kd', 'mhat', 'ds_fl', 'fed_id']:
        logger.info("distillation_epochs: {},\tdistillation_lr: {}".format(args.dis_epochs, args.dis_lr))

    server = Server(args, id=0, model_type=args.central_model, public_dataset=public_dataset)
    clients = {
        k + 1: Client(args, id=k + 1, model_type=args.local_models[k], train_dataset=train_datasets[k]) for k in range(args.K)
    }

    if args.algorithm == 'centralized':
        # ServerExecute():
        merged_train_dataset = concatenate_datasets(train_datasets)
        server.centralized_training(merged_train_dataset)
        # Save server!
        recorder.save_model(server)
        # Eval server!
        recorder.evaluate(server, val_dataset, val_mismatched_dataset)
        # Predict server!
        if args.do_test:
            recorder.predict(server, test_dataset, test_mismatched_dataset, ax_dataset)

        # Save record!
        recorder.save_record()
    elif args.algorithm == 'fed_avg':
        # initialize θ_0
        w_glob = server.model.state_dict()
        for round_ in range(args.R):
            logger.info('===============The {:d}-th round==============='.format(round_ + 1))

            # randomly sample partial clients: m = max(C*K, 1)
            m = max(int(args.C * args.K), 1)
            cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

            # get the quantity of clients joined in the FL train for updating the clients weights
            cur_tot_client_lens = 0
            for k in cur_selected_clients:
                cur_tot_client_lens += len(clients[k].train_dataset)

            w_locals = []
            weights = []
            for k in cur_selected_clients:
                # ClientExecute():
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                # Fork(): θ_t-1^k ← θ_t-1
                client.fork(w_glob)

                """ 1. Local Training """
                # θ_t^k ← ClientUpdate(D^k; θ_t-1^k)
                client.local_update()
                # Save each client!
                recorder.save_model(client)
                # Eval each client!
                recorder.evaluate(client, val_dataset, val_mismatched_dataset)
                # Predict each client!
                if round_ == args.R - 1:  # do prediction for the test_set on the last round
                    if args.do_test:
                        recorder.predict(client, test_dataset, test_mismatched_dataset, ax_dataset)

                """ 2. Upload """
                # Upload local updates
                w_locals.append(client.model.state_dict())
                weights.append(len(client.train_dataset) / cur_tot_client_lens)

            # ServerExecute():
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ 3. Aggregation """
            # Merge(): θ_t ← ∑ (|D^k| / |D|) * θ_t^k
            w_glob = server.merge(w_locals, weights)
            server.model.load_state_dict(w_glob)
            # Save server!
            recorder.save_model(server)
            # Eval server!
            recorder.evaluate(server, val_dataset, val_mismatched_dataset)
            # Predict server!
            if round_ == args.R - 1:  # do prediction for the test_set on the last round
                if args.do_test:
                    recorder.predict(server, test_dataset, test_mismatched_dataset, ax_dataset)

            # Save record!
            recorder.save_record()
    elif args.algorithm == 'fed_df':
        # initialize θ_0
        w_glob = server.model.state_dict()
        for round_ in range(args.R):
            logger.info('===============The {:d}-th round==============='.format(round_ + 1))

            # randomly sample partial clients: m = max(C*K, 1)
            m = max(int(args.C * args.K), 1)
            cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

            # get the quantity of clients joined in the FL train for updating the clients weights
            cur_tot_client_lens = 0
            for k in cur_selected_clients:
                cur_tot_client_lens += len(clients[k].train_dataset)

            w_locals = []
            weights = []
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                # Fork(): θ_t-1^k ← θ_t-1
                client.fork(w_glob)

                """ 1. Local Training """
                # θ_t^k ← ClientUpdate(D^k; θ_t-1^k)
                client.local_update()
                # Save each client!
                recorder.save_model(client)
                # Eval each client!
                recorder.evaluate(client, val_dataset, val_mismatched_dataset)
                # Predict each client!
                if round_ == args.R - 1:  # do prediction for the test_set on the last round
                    if args.do_test:
                        recorder.predict(client, test_dataset, test_mismatched_dataset, ax_dataset)

                """ 2. Upload """
                # Upload local updates
                w_locals.append(client.model.state_dict())
                weights.append(len(client.train_dataset) / cur_tot_client_lens)

            # ServerExecute()
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ 3. Aggregation """
            # Merge(): θ_t,0 ← ∑ (|D^k| / |D|) * θ_t^k
            w_glob = server.merge(w_locals, weights)
            server.model.load_state_dict(w_glob)

            """ 4. Server-side Distillation """
            # Compute local logits
            logits_locals = []
            for k, w_local in zip(cur_selected_clients, w_locals):
                client = clients[k]
                # logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))
                client.model.load_state_dict(w_local)
                logits = client.compute_logits(public_dataset)
                logits_locals.append(logits)
            logits_glob = server.logit_ensemble(logits_locals, weights)
            # Server-side model fusion: θ_t,N ← EnsembleDistillation((X^0, Y_t); θ_t,0)
            server.ensemble_distillation(public_dataset, logits_glob)
            # Save server!
            recorder.save_model(server)
            # Eval server!
            recorder.evaluate(server, val_dataset, val_mismatched_dataset)
            # Predict server!
            if round_ == args.R - 1:  # do prediction for the test_set on the last round
                if args.do_test:
                    recorder.predict(server, test_dataset, test_mismatched_dataset, ax_dataset)

            # Distribute w_t: w_t ← w_t,N
            w_glob = server.model.state_dict()

            # Save record!
            recorder.save_record()
    elif args.algorithm == 'fed_ed':
        # initialize θ_0
        w_glob = server.model.state_dict()
        for round_ in range(args.R):
            logger.info('===============The {:d}-th round==============='.format(round_ + 1))

            # randomly sample partial clients: m = max(C*K, 1)
            m = max(int(args.C * args.K), 1)
            cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

            # get the quantity of clients joined in the FL train for updating the clients weights
            cur_tot_client_lens = 0
            for k in cur_selected_clients:
                cur_tot_client_lens += len(clients[k].train_dataset)  # 498+498+498+498+498=2490

            logits_locals = []
            weights = []
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                # Fork(): θ_t-1^k ← θ_t-1
                client.fork(w_glob)

                """ 1. Local Training """
                # θ_t^k ← ClientUpdate(D^k; θ_t-1^k)
                client.local_update()
                # Save each client!
                recorder.save_model(client)
                # Eval each client!
                recorder.evaluate(client, val_dataset, val_mismatched_dataset)
                # Predict each client!
                if round_ == args.R - 1:  # do prediction for the test_set on the last round
                    if args.do_test:
                        recorder.predict(client, test_dataset, test_mismatched_dataset, ax_dataset)

                """ 2. Local Prediction """
                # Compute local logits
                logits = client.compute_logits(public_dataset)

                """ 3. Upload """
                # Upload local logits
                logits_locals.append(logits)
                weights.append(len(client.train_dataset) / cur_tot_client_lens)

            # ServerExecute()
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ 4. Aggregation """
            logits_glob = server.logit_ensemble(logits_locals, weights)

            """ 5. Distillation """
            # w_t ← EnsembleDistillation((X^0, Y^0) ∪ Y_t; w_t-1)
            server.ensemble_distillation(public_dataset, logits_glob)
            # Save server!
            recorder.save_model(server)
            # Eval server!
            recorder.evaluate(server, val_dataset, val_mismatched_dataset)
            # Predict server!
            if round_ == args.R - 1:  # do prediction for the test_set on the last round
                if args.do_test:
                    recorder.predict(server, test_dataset, test_mismatched_dataset, ax_dataset)

            """ 6. Broadcast """
            # Distribute w_t
            w_glob = server.model.state_dict()

            # Save record!
            recorder.save_record()
    elif args.algorithm == 'fed_kd':
        # for round_ in range(args.R):
        #     logger.info('===============The {:d}-th round==============='.format(round_ + 1))

        # randomly sample partial clients: m = max(C*K, 1)
        m = max(int(args.C * args.K), 1)
        cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

        # get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_client_lens = 0
        for k in cur_selected_clients:
            cur_tot_client_lens += len(clients[k].train_dataset)  # 498+498+498+498+498=2490

        logits_locals = []
        weights = []
        for k in cur_selected_clients:
            # ClientExecute()
            client = clients[k]
            logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

            """ 1. Local Training """
            # θ_t^k ← ClientUpdate(D^k; θ_t-1^k)
            client.local_update()
            # Save each client!
            recorder.save_model(client)
            # Eval each client!
            recorder.evaluate(client, val_dataset, val_mismatched_dataset)
            # Predict each client!
            # if round_ == args.R - 1:  # do prediction for the test_set on the last round
            if args.do_test:
                recorder.predict(client, test_dataset, test_mismatched_dataset, ax_dataset)

            """ 2. Local Prediction """
            # Compute local logits
            logits = client.compute_logits(public_dataset)

            """ 3. Upload """
            # Upload local logits
            logits_locals.append(logits)
            weights.append(len(client.train_dataset) / cur_tot_client_lens)

        # ServerExecute()
        logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

        """ 4. Aggregation """
        logits_glob = server.logit_ensemble(logits_locals, weights)

        """ 5. Distillation """
        # θ_t ← EnsembleDistillation((X^0, Y_t); θ_t-1)
        server.ensemble_distillation(public_dataset, logits_glob)
        # Save server!
        recorder.save_model(server)
        # Eval server!
        recorder.evaluate(server, val_dataset, val_mismatched_dataset)
        # Predict server!
        # if round_ == args.R - 1:  # do prediction for the test_set on the last round
        if args.do_test:
            recorder.predict(server, test_dataset, test_mismatched_dataset, ax_dataset)

        # Save record!
        recorder.save_record()
    elif args.algorithm == 'mhat':
        for round_ in range(args.R):
            logger.info('===============The {:d}-th round==============='.format(round_ + 1))

            # randomly sample partial clients: m = max(C*K, 1)
            m = max(int(args.C * args.K), 1)
            cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

            # get the quantity of clients joined in the FL train for updating the clients weights
            cur_tot_client_lens = 0
            for k in cur_selected_clients:
                cur_tot_client_lens += len(clients[k].train_dataset)  # 498+498+498+498+498=2490

            logits_locals = []
            weights = []
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                """ 1. Local Training """
                # θ_t^k ← ClientUpdate(D^k; θ_t-1^k)
                client.local_update()
                # Save each client!
                recorder.save_model(client)
                # Eval each client!
                recorder.evaluate(client, val_dataset, val_mismatched_dataset)
                # # Predict each client!
                # if round_ == args.R - 1:  # do prediction for the test_set on the last round
                #     if args.do_test:
                #         recorder.predict(client, test_dataset, test_mismatched_dataset, ax_dataset)

                """ 2. Local Prediction """
                # Compute local logits: Y_t^k ← f^k(X^0; θ_t^k)
                logits = client.compute_logits(public_dataset)

                """ 3. Upload """
                # Upload local logits
                logits_locals.append(logits)
                weights.append(len(client.train_dataset) / cur_tot_client_lens)

            # ServerExecute()
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ 4. Server Distillation """
            # ①server distillation: θ_t ← MhatDistillation(D^0 ∪ {Y_t^1, Y_t^2, Y_t^3, Y_t^4, Y_t^5}; θ_t-1)
            server.mhat_distillation(public_dataset, logits_locals, weights)  # aggregated by the server model
            # Save server!
            recorder.save_model(server)
            # Eval server!
            recorder.evaluate(server, val_dataset, val_mismatched_dataset)
            # Predict server!
            if round_ == args.R - 1:  # do prediction for the test_set on the last round
                if args.do_test:
                    recorder.predict(server, test_dataset, test_mismatched_dataset, ax_dataset)

            """ 5. Server Aggregation """
            # Compute server logits as aggregation: Y_t ← f(X^0; θ_t)
            logits_glob = server.compute_logits(public_dataset)

            """ 6. Local Distillation """
            # ②client distillation: θ_t^k ← ClientUpdate((X^0, Y_t); θ_t^k)
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                client.local_distillation(public_dataset, logits_glob)
                # Save each client!
                recorder.save_model(client)
                # Eval each client!
                recorder.evaluate(client, val_dataset, val_mismatched_dataset)
                # Predict each client!
                if round_ == args.R - 1:  # do prediction for the test_set on the last round
                    if args.do_test:
                        recorder.predict(client, test_dataset, test_mismatched_dataset, ax_dataset)

            # Save record!
            recorder.save_record()
    elif args.algorithm == 'ds_fl':
        for round_ in range(args.R):
            logger.info('===============The {:d}-th round==============='.format(round_ + 1))

            # randomly sample partial clients: m = max(C*K, 1)
            m = max(int(args.C * args.K), 1)
            cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

            # get the quantity of clients joined in the FL train for updating the clients weights
            cur_tot_client_lens = 0
            for k in cur_selected_clients:
                cur_tot_client_lens += len(clients[k].train_dataset)

            logits_locals = []
            weights = []
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                """ 1. Local Training """
                # θ_t^k ← ClientUpdate(D^k; θ_t-1^k)
                client.local_update()
                # Save each client!
                recorder.save_model(client)
                # Eval each client!
                recorder.evaluate(client, val_dataset, val_mismatched_dataset)
                # # Predict each client!
                # if round_ == args.R - 1:  # do prediction for the test_set on the last round
                #     if args.do_test:
                #         recorder.predict(client, test_dataset, test_mismatched_dataset, ax_dataset)

                """ 2. Local Prediction """
                # Compute local logits: Y_t^k ← f^k(X^0; θ_t^k)
                logits = client.compute_logits(public_dataset)

                """ 3. Upload """
                # Upload local logits
                logits_locals.append(logits)
                weights.append(len(client.train_dataset) / cur_tot_client_lens)

            # ServerExecute()
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ 4. Aggregation (ERA) """
            logits_glob = server.logit_ensemble_with_ERA(logits_locals, weights)  # softmax(∑ |D^k| / |D| * Y_t^k / t)

            """ 5. Server Distillation """
            # ①server distillation: θ_t ← ServerDistillation((X^0, Y_t); θ_t-1)
            server.ensemble_distillation(public_dataset, logits_glob)
            # Save server!
            recorder.save_model(server)
            # Eval server!
            recorder.evaluate(server, val_dataset, val_mismatched_dataset)
            # Predict server!
            if round_ == args.R - 1:  # do prediction for the test_set on the last round
                if args.do_test:
                    recorder.predict(server, test_dataset, test_mismatched_dataset, ax_dataset)
            """ Local Distillation """
            # ②client distillation: θ_t^k ← ClientDistillation(X^0 ∪ Y_t; θ_t^k)
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                client.local_distillation(public_dataset, logits_glob)
                # Save each client!
                recorder.save_model(client)
                # Eval each client!
                recorder.evaluate(client, val_dataset, val_mismatched_dataset)
                # Predict each client!
                if round_ == args.R - 1:  # do prediction for the test_set on the last round
                    if args.do_test:
                        recorder.predict(client, test_dataset, test_mismatched_dataset, ax_dataset)

            # Save record!
            recorder.save_record()
    elif args.algorithm == 'fed_id':
        for round_ in range(args.R):  # 0~9
            logger.info('===============The {:d}-th round==============='.format(round_ + 1))

            # randomly sample partial clients: m = max(C*K, 1)
            m = max(int(args.C * args.K), 1)
            cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

            # get the quantity of clients joined in the FL train for updating the clients weights
            cur_tot_client_lens = 0
            for k in cur_selected_clients:
                cur_tot_client_lens += len(clients[k].train_dataset)  # 498+498+498+498+498=2490

            """ 1. Local Training """
            weights = []
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                # θ_t^k ← ClientUpdate(θ_t-1^k)
                client.local_update()
                # Save each client!
                recorder.save_model(client)
                # Eval each client!
                recorder.evaluate(client, val_dataset, val_mismatched_dataset)
                # if round_ == args.R - 1:  # do prediction for the test_set on the last round
                #     if args.do_test:
                #         recorder.predict(client, test_dataset, test_mismatched_dataset, ax_dataset)

                weights.append(len(client.train_dataset) / cur_tot_client_lens)

            """ 2. Interactive Distillation """
            # ServerExecute()
            query_dataloader = server.get_query_dataloader(query_dataset)
            query_iter = iter(query_dataloader)

            train_loss = 0.
            for epoch in range(args.dis_epochs):
                logging.info('Epoch {}/{}: '.format(epoch + 1, args.dis_epochs))
                # 将support_dataset分成batch块进行通信和训练
                batch_size = server.batch_size
                num_shard = math.ceil(len(public_dataset) / batch_size)
                for batch_idx in tqdm(range(num_shard), desc='Distilling'):
                    public_batch = public_dataset.shard(num_shard, index=batch_idx)
                    try:
                        query_batch = next(query_iter)
                    except:
                        query_iter = iter(query_dataloader)
                        query_batch = next(query_iter)

                    # 1.Compute `s_loss_l_old`
                    # logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))
                    s_loss_l_old = server.compute_query_loss(query_batch)

                    # 2.Compute `s_loss` to update server

                    batch_logits_locals = []
                    for k in cur_selected_clients:
                        # ClientExecute():
                        client = clients[k]
                        # logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                        batch_logits = client.compute_logits_batch(public_batch)
                        batch_logits_locals.append(batch_logits)

                    batch_logits_glob = server.batch_logit_ensemble(batch_logits_locals, weights)
                    server.batch_distillation(public_batch, batch_logits_locals, weights)

                    # 3.Compute `s_loss_l_new`
                    s_loss_l_new = server.compute_query_loss(query_batch)

                    dot_product = s_loss_l_old - s_loss_l_new

                    # 4.Compute `t_loss` to update locals
                    for i, k in enumerate(cur_selected_clients):
                        # ClientExecute()
                        client = clients[k]
                        # logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                        client.query_update(dot_product, public_batch, batch_logits_locals[i], weights[i], batch_logits_glob)
                        # # Eval each client!
                        # recorder.evaluate(client, val_dataset, val_mismatched_dataset)
                        # # Save each client!
                        # recorder.save_model(client)
                        # # if round_ == args.R - 1:  # do prediction for the test_set on the last round
                        # #     if args.do_test:
                        # #         recorder.predict(client, test_dataset, test_mismatched_dataset, ax_dataset)

                logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))
                # Save server!
                recorder.save_model(server)
                # Eval server!
                recorder.evaluate(server, val_dataset, val_mismatched_dataset)
                # Predict server!
                if round_ == args.R - 1:  # do prediction for the test_set on the last round
                    if args.do_test:
                        recorder.predict(server, test_dataset, test_mismatched_dataset, ax_dataset)

                for k in cur_selected_clients:
                    # ClientExecute()
                    client = clients[k]
                    logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))
                    # Save each client!
                    recorder.save_model(client)
                    # Eval each client!
                    recorder.evaluate(client, val_dataset, val_mismatched_dataset)
                    if round_ == args.R - 1:  # do prediction for the test_set on the last round
                        if args.do_test:
                            recorder.predict(client, test_dataset, test_mismatched_dataset, ax_dataset)

            # Save record!
            recorder.save_record()
    else:
        sys.exit("Not implemented algorithm, code exit, re-run to use correct algorithm")
