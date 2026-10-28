# Flow Matching Disentangle

## 环境

```bash
conda env create -f environment.yaml
conda activate encdiff
```

## 训练

第一阶段得到的 VQ checkpoint 写入对应主实验配置的 `ckpt_path`。当前三份配置分别指向 `pretrained_models/shapes3d.ckpt`、`pretrained_models/mpi3d.ckpt`、`pretrained_models/cars3d.ckpt`。

### Shapes3D

```bash
python main.py -b configs/autoencoder/shapes3d_vq_4_16.yaml -t --gpus 0,
python main_val.py -b configs/latent-diffusion/shapes3d.yaml -t --gpus 0,
```

### MPI3D

```bash
python main.py -b configs/autoencoder/mpi3d_vq_4_16.yaml -t --gpus 0,
python main_val.py -b configs/latent-diffusion/mpi3d.yaml -t --gpus 0,
```

### Cars3D

```bash
python main.py -b configs/autoencoder/cars3d_vq_4_16.yaml -t --gpus 0,
python main_val.py -b configs/latent-diffusion/cars3d.yaml -t --gpus 0,
```

## 致谢

本仓库的潜空间扩散骨架来自 [LDM](https://github.com/CompVis/latent-diffusion) 与 [EncDiff](https://arxiv.org/pdf/2402.09712)。解耦指标来自 [disentanglement_lib](https://github.com/google-research/disentanglement_lib)。


