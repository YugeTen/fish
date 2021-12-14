import copy
import argparse
import datetime
import json
import os
import sys
import csv
import tqdm
from collections import defaultdict
from tempfile import mkdtemp

import numpy as np
import torch
import torch.optim as optim

import models
from config import dataset_defaults
from utils import unpack_data, sample_domains, save_best_model, \
    Logger, return_predict_fn, return_criterion, fish_step

runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Gradient Matching for Domain Generalization.')
# General
parser.add_argument('--dataset', type=str,
                    help="Name of dataset, choose from amazon, camelyon, "
                         "cdsprites, civil, fmow, iwildcam, poverty")
parser.add_argument('--algorithm', type=str,
                    help='training scheme, choose between fish or erm.')
parser.add_argument('--experiment', type=str, default='.',
                    help='experiment name, set as . for automatic naming.')
parser.add_argument('--data-dir', type=str,
                    help='path to data dir')
parser.add_argument('--stratified', action='store_true', default=False,
                    help='whether to use stratified sampling for classes')
parser.add_argument('--num-domains', type=int, default=15,
                    help='Number of domains, only specify for cdsprites')
# Computation
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA use')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed, set as -1 for random.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

args_dict = args.__dict__
args_dict.update(dataset_defaults[args.dataset])
args = argparse.Namespace(**args_dict)

# Choosing and saving a random seed for reproducibility
if args.seed == -1:
    args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# experiment directory setup
args.experiment = f"{args.dataset}_{args.algorithm}" \
    if args.experiment == '.' else args.experiment
directory_name = '../experiments/{}'.format(args.experiment)
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
runPath = mkdtemp(prefix=runId, dir=directory_name)

# logging setup
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:' + runPath)
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, '{}/args.rar'.format(runPath))

# load model
modelC = getattr(models, args.dataset)
train_loader, tv_loaders = modelC.getDataLoaders(args, device=device)
val_loader, test_loader = tv_loaders['val'], tv_loaders['test']
model = modelC(args, weights=None).to(device)

assert args.optimiser in ['SGD', 'Adam'], "Invalid choice of optimiser, choose between 'Adam' and 'SGD'"
opt = getattr(optim, args.optimiser)
optimiserC = opt(model.parameters(), **args.optimiser_args)
predict_fn, criterion = return_predict_fn(args.dataset), return_criterion(args.dataset)

def pretrain(train_loader, pretrain_iters):
    aggP = defaultdict(list)
    aggP['val_stat'] = [0.]

    n_iters = 0
    pretrain_epochs = int(np.ceil(pretrain_iters/len(train_loader)))
    pbar = tqdm.tqdm(total = pretrain_iters)
    for epoch in range(pretrain_epochs):
        for i, data in enumerate(train_loader):
            model.train()
            # get the inputs
            x, y = unpack_data(data, device)
            optimiserC.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimiserC.step()
            n_iters += 1
            # display progress
            pbar.set_description(f"Pretrain {n_iters}/{pretrain_iters} iters")
            pbar.update(1)
            if (i + 1) % args.print_iters == 0 and args.print_iters != -1:
                test(val_loader, aggP, loader_type='val', verbose=False)
                test(test_loader, aggP, loader_type='test', verbose=False)
                save_best_model(model, runPath, aggP, args)

            if n_iters == pretrain_iters:
                test(val_loader, aggP, loader_type='val', verbose=False)
                test(test_loader, aggP, loader_type='test', verbose=False)
                save_best_model(model, runPath, aggP, args)
                break
    pbar.close()

    model.load_state_dict(torch.load(runPath + '/model.rar'))
    print('Finished ERM pre-training!')

def train_erm(train_loader, epoch, agg):
    running_loss = 0
    total_iters = len(train_loader)
    print('\n====> Epoch: {:03d} '.format(epoch))
    for i, data in enumerate(train_loader):
        model.train()
        # get the inputs
        x, y = unpack_data(data, device)
        optimiserC.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimiserC.step()
        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and args.algorithm != 'fish':
            agg['train_loss'].append(running_loss / args.print_iters)
            agg['train_iters'].append(i+1+epoch*total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_iters))
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            running_loss = 0.0
            save_best_model(model, runPath, agg, args)


def train_fish(train_loader, epoch, agg):
    model.train()
    train_loader.dataset.reset_batch()
    i = 0
    print('\n====> Epoch: {:03d} '.format(epoch))
    opt_inner_pre = None

    while sum([l > 1 for l in train_loader.dataset.batches_left.values()]) >= args.meta_steps:
        i += 1
        # sample `meta_steps` number of domains to use for the inner loop
        domains = sample_domains(train_loader, args.meta_steps, args.stratified).tolist()
        # prepare model for inner loop update
        model_inner = copy.deepcopy(model)
        model_inner.train()
        opt_inner = opt(model_inner.parameters(), **args.optimiser_args)
        if opt_inner_pre is not None and args.reload_inner_optim:
            opt_inner.load_state_dict(opt_inner_pre)
        # inner loop update
        for domain in domains:
            data = train_loader.dataset.get_batch(domain)
            x, y = unpack_data(data, device)
            opt_inner.zero_grad()
            y_hat = model_inner(x)
            loss = criterion(y_hat, y)
            loss.backward()
            opt_inner.step()
        opt_inner_pre = opt_inner.state_dict()
        # fish update
        meta_weights = fish_step(meta_weights=model.state_dict(),
                                 inner_weights=model_inner.state_dict(),
                                 meta_lr=args.meta_lr / args.meta_steps)
        model.reset_weights(meta_weights)
        # log the number of batches left for each domain
        for domain in domains:
            train_loader.dataset.batches_left[domain] = \
                train_loader.dataset.batches_left[domain] - 1 \
                if train_loader.dataset.batches_left[domain] > 1 else 1

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1:
            print(f'iteration {(i + 1):05d}: ')
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            model.train()
            save_best_model(model, runPath, agg, args)


def test(test_loader, agg, loader_type='test', verbose=True, save_ypred=False):
    model.eval()
    yhats, ys, metas = [], [], []
    with torch.no_grad():
        for i, (x, y, meta) in enumerate(test_loader):
            # get the inputs
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            ys.append(y)
            yhats.append(y_hat)
            metas.append(meta)

        ypreds, ys, metas = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(metas)
        if save_ypred:
            if args.dataset == 'poverty':
                save_name = f"{args.dataset}_split:{loader_type}_fold:" \
                            f"{['A', 'B', 'C', 'D', 'E'][args.seed]}" \
                            f"_epoch:best_pred.csv"
            else:
                save_name = f"{args.dataset}_split:{loader_type}_seed:" \
                            f"{args.seed}_epoch:best_pred.csv"
            with open(f"{runPath}/{save_name}", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ypreds.unsqueeze(1).cpu().tolist())
        test_val = test_loader.dataset.eval(ypreds.cpu(), ys.cpu(), metas)
        agg[f'{loader_type}_stat'].append(test_val[0][args.selection_metric])
        if verbose:
            print(f"=============== {loader_type} ===============\n{test_val[-1]}")


if __name__ == '__main__':
    if args.algorithm == 'fish' and args.pretrain_iters != 0:
        print("="*30 + "ERM pretrain" + "="*30)
        pretrain(train_loader, args.pretrain_iters)

    print("="*30 + f"Training: {args.algorithm}" + "="*30)
    train = locals()[f'train_{args.algorithm}']
    agg = defaultdict(list)
    agg['val_stat'] = [0.]
    for epoch in range(args.epochs):
        train(train_loader, epoch, agg)
        test(val_loader, agg, loader_type='val')
        test(test_loader, agg, loader_type='test')
        save_best_model(model, runPath, agg, args)

    model.load_state_dict(torch.load(runPath + '/model.rar'))
    print('Finished training! Loading best model...')
    for split, loader in tv_loaders.items():
        test(loader, agg, loader_type=split, save_ypred=True)
