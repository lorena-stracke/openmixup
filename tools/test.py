import argparse
import importlib
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from openmixup.datasets import build_dataloader, build_dataset
from openmixup.models import build_model
from openmixup.utils import (get_root_logger, dist_forward_collect, 
                             setup_multi_processes, nondist_forward_collect, traverse_replace)


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
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


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

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader)  # dict{key: np.ndarray}

    rank, _ = get_dist_info()
    if rank == 0:
        for name, val in outputs.items():
            dataset.evaluate(
                torch.from_numpy(val), keyword=name, logger=logger,
                **cfg.evaluation.get('eval_param', dict(topk=(1, 5))))


if __name__ == '__main__':
    main()
