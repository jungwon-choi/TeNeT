"""Train and evaluate TeNeT with k-fold cross-validation.

Single GPU:
    python train.py --config configs/hcpep_dx.yaml --data_dir ./data --output_dir ./result/hcpep_dx
Multiple GPUs:
    torchrun --nproc_per_node 2 train.py --config configs/hcpep_dx.yaml --data_dir ./data --output_dir ./result/hcpep_dx
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from tenet.config import Config, load_config, add_override_arguments
from tenet.data import FMRIDataset
from tenet.model import TeNeT
from tenet.metrics import evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id, seed):
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def parse_args():
    parser = argparse.ArgumentParser(description='TeNeT training', allow_abbrev=False)
    parser.add_argument('--config', required=True, help='experiment config (configs/<dataset>_<task>.yaml)')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--folds', nargs='+', default=['all'], help="fold indices, or 'all'")
    parser.add_argument('--no_progress', action='store_true')
    args, rest = parser.parse_known_args()

    cfg = load_config(args.config)
    override = argparse.ArgumentParser(allow_abbrev=False)
    add_override_arguments(override, cfg)
    for key, value in vars(override.parse_args(rest)).items():
        if value is not None:
            cfg[key] = value
    return args, Config(cfg)


def window_starts(cfg):
    # windows are taken from the first `dynamic_length` time points (also at test time)
    return list(range(0, cfg.dynamic_length - cfg.window_size, cfg.window_stride))


def gather(tensor):
    if not dist.is_initialized():
        return tensor
    out = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(out, tensor.contiguous())
    return torch.cat(out)


def run_epoch(model, loader, dataset, cfg, fold, mode, device, optimizer=None, scheduler=None, epoch=0, progress=True):
    """One pass over the train/val/test split. Returns loss, metrics, and predictions (rank 0)."""
    train = mode == 'train'
    model.train(train)
    dataset.set_fold(fold, mode=mode)
    criterion = torch.nn.MSELoss() if dataset.num_classes == 1 else torch.nn.CrossEntropyLoss()
    starts = window_starts(cfg)

    losses, trues, logits, ids = [], [], [], []
    with torch.set_grad_enabled(train):
        for i, batch in enumerate(tqdm(loader, desc=f'{mode} {epoch}', leave=False, disable=not progress)):
            label = batch['label'].to(device)
            logit, _, _ = model(batch['timeseries'].to(device), starts, cfg.window_size)
            if dataset.num_classes == 1 and logit.dim() == 2:
                logit = logit.squeeze(-1)
            loss = criterion(logit, label)

            if train:
                (loss / cfg.gradient_accumulation_steps).backward()
                if (i + 1) % cfg.gradient_accumulation_steps == 0:
                    if cfg.clip_grad > 0:
                        torch.nn.utils.clip_grad_value_(model.parameters(), cfg.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()
                scheduler.step()

            losses.append(loss.detach())
            trues.append(gather(label.detach()))
            logits.append(gather(logit.detach().float()))
            ids.extend(batch['id'])

    loss = torch.stack(losses).mean()
    if dist.is_initialized() and train:
        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    true, logit = torch.cat(trues).cpu(), torch.cat(logits).cpu()
    if dataset.num_classes == 1:
        true = dataset.label_encoder.inverse_transform(true).numpy()
        pred = prob = dataset.label_encoder.inverse_transform(logit).numpy()
    else:
        prob = logit.softmax(1).numpy()
        pred, true = prob.argmax(1), true.numpy()

    results = {'epoch': epoch, 'split': mode, 'loss': loss.item(), **evaluate(true, pred, prob, dataset.num_classes)}
    return results, {'id': np.array(ids), 'true': true, 'pred': pred, 'prob': prob}


def train_fold(cfg, args, fold, rank, world_size, device):
    out = Path(args.output_dir) / f'fold{fold}'
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    dataset = FMRIDataset(args.data_dir, cfg.dataset, cfg.roi, cfg.target_feature, k_fold=cfg.k_fold,
                          regression=cfg.regression, dynamic_length=cfg.dynamic_length, val_ratio=cfg.val_ratio)
    dataset.set_fold(fold, mode='train', verbose=rank == 0)

    if cfg.minibatch_size % world_size != 0:
        raise ValueError(f'minibatch_size ({cfg.minibatch_size}) must be divisible by the number of GPUs ({world_size})')
    sampler = DistributedSampler(dataset, shuffle=True, seed=cfg.seed) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=cfg.minibatch_size // world_size, sampler=sampler, drop_last=True,
                        num_workers=cfg.num_workers, pin_memory=True,
                        worker_init_fn=lambda w: seed_worker(w, cfg.seed),
                        generator=torch.Generator().manual_seed(cfg.seed))
    eval_loader = DataLoader(dataset, batch_size=cfg.eval_minibatch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    model = TeNeT(dataset.num_nodes, dataset.num_classes, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
                  num_time_layers=cfg.num_time_layers, num_heads=cfg.num_heads, dropout=cfg.dropout,
                  cls_token=cfg.cls_token).to(device)
    if rank == 0:
        logging.info(f'parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    net = DDP(model, device_ids=[device], find_unused_parameters=True) if world_size > 1 else model

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr * 2, epochs=cfg.num_epochs, steps_per_epoch=len(loader),
        pct_start=0.2, div_factor=2, final_div_factor=1000)

    log = (out / 'train_log.jsonl').open('w') if rank == 0 else None
    best = None
    for epoch in range(cfg.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        results, _ = run_epoch(net, loader, dataset, cfg, fold, 'train', device, optimizer, scheduler, epoch,
                               progress=not args.no_progress and rank == 0)
        if rank == 0:
            logging.info(json.dumps(results))
            log.write(json.dumps(results) + '\n')

        last = epoch + 1 == cfg.num_epochs
        if rank == 0 and ((epoch + 1) % cfg.eval_every == 0 or last):
            torch.save(model.state_dict(), out / 'model.pth')
            if cfg.val_ratio > 0:
                val, _ = run_epoch(model, eval_loader, dataset, cfg, fold, 'val', device, epoch=epoch, progress=False)
                logging.info(json.dumps(val))
                log.write(json.dumps(val) + '\n')
                score = -val['mse'] if dataset.num_classes == 1 else val['roc_auc']
                if best is None or score >= best:
                    best = score
                    torch.save(model.state_dict(), out / 'best_model.pth')
            test, preds = run_epoch(model, eval_loader, dataset, cfg, fold, 'test', device, epoch=epoch, progress=False)
            logging.info(json.dumps(test))
            log.write(json.dumps(test) + '\n')
            log.flush()
            if last:
                np.savez(out / 'test_predictions.npz', **preds)
                with (out / 'test_metrics.json').open('w') as f:
                    json.dump(test, f, indent=1)
        if world_size > 1:
            dist.barrier()
    if log:
        log.close()


def summarize(output_dir, folds):
    rows = []
    for k in folds:
        path = Path(output_dir) / f'fold{k}' / 'test_metrics.json'
        if path.exists():
            rows.append(json.load(path.open()))
    if not rows:
        return
    keys = [k for k in rows[0] if k not in ('epoch', 'split', 'loss')]
    lines = ['metric,' + ','.join(f'fold{k}' for k in folds) + ',mean,std']
    for key in keys:
        values = [r[key] for r in rows]
        lines.append(f'{key},' + ','.join(f'{v:.4f}' for v in values) + f',{np.mean(values):.4f},{np.std(values):.4f}')
    (Path(output_dir) / 'results.csv').write_text('\n'.join(lines) + '\n')
    logging.info('\n' + '\n'.join(lines))


def main():
    args, cfg = parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('LOCAL_RANK', 0))
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)] +
                        ([logging.FileHandler(Path(args.output_dir) / 'train.log')] if rank == 0 else []))
    if rank == 0:
        with (Path(args.output_dir) / 'config.json').open('w') as f:
            json.dump(cfg, f, indent=1)

    folds = list(range(cfg.k_fold)) if args.folds == ['all'] else [int(k) for k in args.folds]
    for fold in folds:
        if rank == 0:
            logging.info(f'===== fold {fold} =====')
        train_fold(cfg, args, fold, rank, world_size, device)
    if rank == 0:
        summarize(args.output_dir, folds)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
