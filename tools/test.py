import argparse
import copy
import csv
import importlib
import os
import os.path as osp
import random
import time

import mmcv
import numpy as np
import torch
from PIL import Image
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from openmixup.datasets import build_dataloader, build_dataset
from openmixup.datasets.registry import PIPELINES
from openmixup.models import build_model
from openmixup.utils import (get_root_logger, build_from_cfg, dist_forward_collect,
                             setup_multi_processes, nondist_forward_collect,
                             traverse_replace)


TENSOR_TRANSFORMS = {'ToTensor', 'ImageToTensor', 'Normalize', 'Normalize_mmcls'}


def single_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    results = nondist_forward_collect(func, data_loader,
                                      len(data_loader.dataset))
    return results


def multi_gpu_test(model, data_loader):
    model.eval()
    func = lambda **x: model(mode='test', **x)
    rank, world_size = get_dist_info()
    results = dist_forward_collect(func, data_loader, rank,
                                   len(data_loader.dataset))
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--local_rank',
        help='set local_rank for torch.distributed.launch (torch<2.0.0)',
        type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500,
        help='port only works when launcher=="slurm"')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--blur', action='store_true', help='Enable blur preprocessing (default_type: black_white contrast')
    parser.add_argument('--blur-depth', dest="blur_depth", type=int, default=5, help='Depth for blur contrast')
    parser.add_argument('--single-color', dest="single_color", action='store_true', help='Enable single-channel contrast')
    parser.add_argument('--color-opponency', dest="color_opponency", action='store_true', help='Enable color-opponency contrast')
    parser.add_argument('--black-white', dest="black_white", action='store_true', help='Enable blur preprocessing (default_type: False')
    parser.add_argument('--normalize', dest="normalize", action='store_true', help='Enable normalization after the preprocessing (default_type: False')
    parser.add_argument('--channels', type=int, default=3, help='Channels during preprocessing')
    parser.add_argument('--sparsity-threshold', type=float, default=0.0, help='Threshold in which range the contrast images are set to 0.0')
    parser.add_argument('--sparsity-type', type=str, choices=['threshold', 'percentage'], default='threshold', help="Sparsity type: 'threshold' or 'percentage'")
    parser.add_argument('--change-range', action='store_true', help='Change range of all channels to the one from the first channel')
    parser.add_argument('--sparse-baseline', action='store_true', help='Whether to test the baseline with sparsity (without any blur preprocessing)')
    parser.add_argument('--use-reflect-padding-for-blurring', action='store_true', help='Whether to use reflect padding for blurring instead of zero padding (default: False)')
    parser.add_argument(
        '--eval-corruptions',
        action='store_true',
        help='Evaluate clean CIFAR plus common imagecorruptions corruptions.')
    parser.add_argument(
        '--corruption-csv',
        type=str,
        default=None,
        help='CSV path for corruption results. Defaults to work_dir.')
    parser.add_argument(
        '--corruption-names',
        nargs='+',
        default=None,
        help='Corruption names to evaluate. Defaults to the 15 common corruptions.')
    parser.add_argument(
        '--corruption-severities',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 5],
        help='Severity levels to evaluate.')
    parser.add_argument(
        '--corruption-insert-index',
        type=int,
        default=0,
        help='Pipeline index for ImageCorruption. Use -1 to insert before ToTensor.')
    parser.add_argument(
        '--max-eval-samples',
        type=int,
        default=None,
        help='Limit evaluation samples for smoke tests.')
    parser.add_argument(
        '--save-corruption-samples',
        action='store_true',
        help='Save clean/severity image strips for visual inspection.')
    parser.add_argument(
        '--corruption-samples-dir',
        type=str,
        default=None,
        help='Directory for saved corruption sample images.')
    parser.add_argument(
        '--corruption-sample-index',
        type=int,
        default=0,
        help='Dataset index used for saved corruption samples.')
    parser.add_argument(
        '--corruption-sample-scale',
        type=int,
        default=4,
        help='Nearest-neighbor scale for saved CIFAR corruption samples.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed for stochastic image corruptions.')
    args = parser.parse_args()
    args.corruption_seed = args.seed
    
    # Set manual seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def get_common_corruption_names():
    try:
        from imagecorruptions import get_corruption_names
    except ImportError as exc:
        raise ImportError(
            'Corruption evaluation requires imagecorruptions. Install it with '
            '`pip install imagecorruptions`.') from exc
    return list(get_corruption_names('common'))


def insert_corruption_transform(data_cfg, corruption_name, severity, insert_index):
    data_cfg = copy.deepcopy(data_cfg)
    pipeline = list(data_cfg.pipeline)
    transform = dict(
        type='ImageCorruption',
        corruption_name=corruption_name,
        severity=severity)

    if insert_index < 0:
        insert_at = len(pipeline)
        for i, step in enumerate(pipeline):
            step_type = step.get('type') if isinstance(step, dict) else None
            if step_type in TENSOR_TRANSFORMS:
                insert_at = i
                break
    else:
        insert_at = min(insert_index, len(pipeline))

    pipeline.insert(insert_at, transform)
    data_cfg.pipeline = pipeline
    return data_cfg


def limit_cifar_dataset(dataset, max_samples, logger=None):
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError('--max-eval-samples must be positive.')

    num_samples = min(max_samples, len(dataset))
    data_source = getattr(dataset, 'data_source', None)
    cifar = getattr(data_source, 'cifar', None)
    if cifar is None or not hasattr(cifar, 'data') or not hasattr(cifar, 'targets'):
        raise NotImplementedError(
            '--max-eval-samples currently supports CIFAR-style data sources.')

    cifar.data = cifar.data[:num_samples]
    cifar.targets = list(np.asarray(cifar.targets)[:num_samples])
    data_source.labels = cifar.targets
    dataset.targets = np.asarray(cifar.targets)
    if logger is not None:
        logger.info('Limited evaluation dataset to %d samples.', num_samples)
    return dataset


def build_eval_dataset(data_cfg, max_eval_samples=None, logger=None):
    dataset = build_dataset(data_cfg)
    return limit_cifar_dataset(dataset, max_eval_samples, logger=logger)


def run_test(model,
             data_cfg,
             cfg,
             distributed,
             logger,
             max_eval_samples=None,
             dataloader_seed=None):
    if dataloader_seed is not None:
        np.random.seed(dataloader_seed)
        random.seed(dataloader_seed)
        torch.manual_seed(dataloader_seed)

    dataset = build_eval_dataset(data_cfg, max_eval_samples, logger=logger)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        seed=dataloader_seed)

    if not distributed:
        outputs = single_gpu_test(model, data_loader)
    else:
        outputs = multi_gpu_test(model, data_loader)

    rank, _ = get_dist_info()
    eval_results = {}
    if rank == 0:
        for name, val in outputs.items():
            eval_results.update(
                dataset.evaluate(
                    torch.from_numpy(val),
                    keyword=name,
                    logger=logger,
                    **cfg.evaluation.get('eval_param', dict(topk=(1, 5)))))
    return eval_results


def first_metric(eval_results, suffix):
    for key in sorted(eval_results):
        if key.endswith(suffix):
            return eval_results[key]
    return ''


def make_result_row(args, cfg, corruption, severity, summary, eval_results):
    data_source = cfg.data.val.get('data_source', {})
    row = dict(
        config=args.config,
        checkpoint=args.checkpoint,
        dataset=data_source.get('type', ''),
        dataset_split=data_source.get('split', ''),
        corruption=corruption,
        severity=severity,
        summary=summary,
        top1=first_metric(eval_results, '_top1'),
        top5=first_metric(eval_results, '_top5'))
    row.update(eval_results)
    return row


def add_mean_rows(rows, corruptions):
    non_metric_keys = {
        'config', 'checkpoint', 'dataset', 'dataset_split', 'corruption',
        'severity', 'summary'
    }
    metric_keys = sorted({
        key
        for row in rows
        for key, value in row.items()
        if key not in non_metric_keys
        if isinstance(value, (float, int, np.floating, np.integer))
    })

    for corruption in corruptions:
        severity_rows = [
            row for row in rows
            if row['summary'] == 'severity' and row['corruption'] == corruption
        ]
        if severity_rows:
            mean_row = copy.deepcopy(severity_rows[0])
            mean_row['severity'] = 'mean'
            mean_row['summary'] = 'corruption_mean'
            for key in metric_keys:
                values = [float(row[key]) for row in severity_rows if row.get(key) != '']
                if values:
                    mean_row[key] = sum(values) / len(values)
            rows.append(mean_row)

    all_rows = [row for row in rows if row['summary'] == 'severity']
    if all_rows:
        mean_row = copy.deepcopy(all_rows[0])
        mean_row['corruption'] = 'all'
        mean_row['severity'] = 'mean'
        mean_row['summary'] = 'all_corruptions_mean'
        for key in metric_keys:
            values = [float(row[key]) for row in all_rows if row.get(key) != '']
            if values:
                mean_row[key] = sum(values) / len(values)
        rows.append(mean_row)


def write_results_csv(rows, csv_path):
    mmcv.mkdir_or_exist(osp.dirname(osp.abspath(csv_path)))
    base_fields = [
        'config', 'checkpoint', 'dataset', 'dataset_split', 'corruption',
        'severity', 'summary', 'top1', 'top5'
    ]
    extra_fields = sorted({key for row in rows for key in row} - set(base_fields))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        writer.writerows(rows)


def save_corruption_samples(data_cfg, cfg, args, corruptions, severities, logger):
    rank, _ = get_dist_info()
    if rank != 0:
        return

    dataset = build_dataset(data_cfg)
    if args.corruption_sample_index >= len(dataset):
        raise ValueError('--corruption-sample-index is outside the dataset.')

    img, _ = dataset.data_source.get_sample(args.corruption_sample_index)
    if not isinstance(img, Image.Image):
        raise TypeError('Expected the data source sample to be a PIL image.')

    image = np.array(img.convert('RGB'))
    scale = max(1, args.corruption_sample_scale)
    out_dir = args.corruption_samples_dir or osp.join(
        cfg.work_dir, 'corruption_samples')
    mmcv.mkdir_or_exist(out_dir)

    for corruption_name in corruptions:
        np.random.seed(args.corruption_seed)
        random.seed(args.corruption_seed)
        panels = [Image.fromarray(image)]
        for severity in severities:
            transform = build_from_cfg(
                dict(
                    type='ImageCorruption',
                    corruption_name=corruption_name,
                    severity=severity), PIPELINES)
            panels.append(transform(img))
        if scale > 1:
            panels = [
                panel.resize(
                    (panel.width * scale, panel.height * scale),
                    Image.NEAREST) for panel in panels
            ]

        strip = Image.new(
            'RGB',
            (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
            (255, 255, 255))
        x_offset = 0
        for panel in panels:
            strip.paste(panel, (x_offset, 0))
            x_offset += panel.width
        strip.save(osp.join(out_dir, f'{corruption_name}.png'))
    logger.info('Saved corruption sample strips to %s', out_dir)


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])
    cfg.gpu_ids = [args.gpu_id]

    cfg.model.pretrained = None  # ensure to use checkpoint rather than pretraining

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'test_{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    preprocessing_dict = dict(
        type='BlurPreprocessing',
        blur_bool=args.blur,
        blur_depth=args.blur_depth,
        single_color=args.single_color,
        color_opponency=args.color_opponency,
        channels=args.channels,
        path=args.work_dir,
        training=False,
        black_white=args.black_white,
        normalize=args.normalize,
        sparsity_threshold=args.sparsity_threshold,
        sparsity_type = args.sparsity_type,
        change_range=args.change_range,
        sparse_baseline = args.sparse_baseline,
        use_reflect_padding_for_blurring = args.use_reflect_padding_for_blurring
    )

    if 'preprocessing' not in cfg.model:
        cfg.model.backbone['preprocessing'] = preprocessing_dict
    else:
        cfg.model.backbone['preprocessing'].update(preprocessing_dict)

    # build the model and load checkpoint
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    rank, _ = get_dist_info()
    rows = []

    clean_results = run_test(
        model,
        copy.deepcopy(cfg.data.val),
        cfg,
        distributed,
        logger,
        max_eval_samples=args.max_eval_samples)
    if rank == 0:
        rows.append(
            make_result_row(
                args,
                cfg,
                corruption='clean',
                severity=0,
                summary='clean',
                eval_results=clean_results))

    if args.eval_corruptions:
        corruptions = args.corruption_names or get_common_corruption_names()
        severities = args.corruption_severities
        invalid_severities = sorted(set(severities) - {1, 2, 3, 4, 5})
        if invalid_severities:
            raise ValueError(
                f'Invalid severities {invalid_severities}; expected values in [1, 5].')

        if args.save_corruption_samples:
            save_corruption_samples(
                copy.deepcopy(cfg.data.val),
                cfg,
                args,
                corruptions,
                severities,
                logger)

        for corruption_name in corruptions:
            for severity in severities:
                logger.info(
                    'Evaluating corruption=%s severity=%s',
                    corruption_name,
                    severity)
                corrupted_cfg = insert_corruption_transform(
                    cfg.data.val,
                    corruption_name,
                    severity,
                    args.corruption_insert_index)
                eval_results = run_test(
                    model,
                    corrupted_cfg,
                    cfg,
                    distributed,
                    logger,
                    max_eval_samples=args.max_eval_samples,
                    dataloader_seed=args.corruption_seed)
                if rank == 0:
                    rows.append(
                        make_result_row(
                            args,
                            cfg,
                            corruption=corruption_name,
                            severity=severity,
                            summary='severity',
                            eval_results=eval_results))

        if rank == 0:
            add_mean_rows(rows, corruptions)
            csv_path = args.corruption_csv or osp.join(
                cfg.work_dir, f'corruption_results_{timestamp}.csv')
            write_results_csv(rows, csv_path)
            logger.info('Saved corruption results CSV to %s', csv_path)


if __name__ == '__main__':
    main()
