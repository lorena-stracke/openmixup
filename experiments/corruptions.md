# CIFAR-100-C Common Corruption Evaluation

This repo now supports on-the-fly CIFAR-100-C style testing with the
`imagecorruptions` package. The test command evaluates the clean CIFAR-100 test
split first, then the 15 common corruptions at severities 1 through 5.

## Environment

From the repo root:

```bash
conda create -n openmixup_corruptions python=3.9 -y
conda activate openmixup_corruptions
export PYTHONNOUSERSITE=1

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv_full-1.7.2-cp39-cp39-manylinux1_x86_64.whl
pip install -r requirements/runtime.txt
pip install imagecorruptions scikit-image yapf==0.40.1
pip install -e .
```

Download CIFAR-100 if it is not already under `data/cifar100`:

```bash
python -c "import torchvision; torchvision.datasets.CIFAR100(download=True, root='data/cifar100')"
```

## Run Full Evaluation

Single GPU:

```bash
export PYTHONNOUSERSITE=1
python tools/test.py \
  configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_vanilla_bs100_ep200.py \
  /path/to/checkpoint.pth \
  --work_dir work_dirs/cifar100_corruptions/convnext_t_vanilla \
  --eval-corruptions \
  --save-corruption-samples
```

Distributed:

```bash
export PYTHONNOUSERSITE=1
bash tools/dist_test.sh \
  configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_vanilla_bs100_ep200.py \
  4 \
  /path/to/checkpoint.pth \
  --eval-corruptions \
  --save-corruption-samples
```

The CSV is written to `work_dir/corruption_results_<timestamp>.csv` unless you
pass `--corruption-csv /path/to/results.csv`.

Stochastic corruptions use `--corruption-seed 0` by default for repeatable
evaluation. Pass a different seed if you want a different on-the-fly draw.

The CSV contains:

- one clean row with the normal CIFAR-100 test accuracy;
- one row for each `corruption x severity`;
- one `corruption_mean` row per corruption, averaged over the evaluated
  severities;
- one `all_corruptions_mean` row averaged over all evaluated corruption and
  severity pairs.

## Smoke Test

First run a one-epoch CIFAR-100 training smoke test. The one-process
distributed launcher follows the code path that works cleanly for the CIFAR
configs in this checkout:

```bash
export PYTHONNOUSERSITE=1
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29613 \
  tools/train.py \
  configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_vanilla_bs100_ep200.py \
  --work_dir work_dirs/smoke_cifar_train_convnext_dist \
  --launcher pytorch \
  --seed 0 \
  --cfg-options \
    runner.max_epochs=1 \
    data.imgs_per_gpu=16 \
    data.workers_per_gpu=1 \
    data.train.data_source.num_labeled=100 \
    evaluation.interval=999 \
    checkpoint_config.interval=1 \
    log_config.interval=1
```

Then use a small sample and two corruptions to confirm the test pipeline
quickly:

```bash
export PYTHONNOUSERSITE=1
python tools/test.py \
  configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_vanilla_bs100_ep200.py \
  work_dirs/smoke_cifar_train_convnext_dist/epoch_1.pth \
  --work_dir work_dirs/smoke_cifar_corruptions \
  --eval-corruptions \
  --corruption-names gaussian_noise brightness \
  --corruption-severities 1 5 \
  --max-eval-samples 128 \
  --save-corruption-samples
```

For a slightly broader smoke test that touches all 15 common corruptions while
still staying small:

```bash
export PYTHONNOUSERSITE=1
python tools/test.py \
  configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_vanilla_bs100_ep200.py \
  work_dirs/smoke_cifar_train_convnext_dist/epoch_1.pth \
  --work_dir work_dirs/smoke_cifar_corruptions_all15 \
  --eval-corruptions \
  --max-eval-samples 16 \
  --save-corruption-samples \
  --cfg-options data.imgs_per_gpu=16 data.workers_per_gpu=1
```

The saved sample strips are in `work_dir/corruption_samples/`. Each strip is
`clean, severity 1, severity 2, ...` for the selected severities. The corruption
transform is inserted at the start of the test pipeline by default, so CIFAR
images are corrupted as uint8 PIL/NumPy images before `ToTensor` and
normalization. Use `--corruption-insert-index -1` to insert immediately before
tensor conversion instead.
