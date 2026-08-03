# VadCLIP
This is the official Pytorch implementation of our paper:
**"VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection"** in **AAAI 2024.**  
> <a href="https://scholar.google.com.hk/citations?user=QkNqUH4AAAAJ" target="_blank">Peng Wu</a>, <a href="https://scholar.google.com/citations?user=ljzQLv4AAAAJ" target="_blank">Xuerong Zhou</a>, <a href="https://scholar.google.com.hk/citations?hl=zh-CN&user=1ZO7pHkAAAAJ" target="_blank">Guansong Pang</a>, <a href="https://paperswithcode.com/search?q=author%3ALingru+Zhou" target="_blank">Lingru Zhou</a>,  <a href="https://scholar.google.com/citations?user=BSGy3foAAAAJ" target="_blank">Qingsen Yan</a>, <a href="https://scholar.google.com.au/citations?user=aPLp7pAAAAAJ" target="_blank">Peng Wang</a>, <a href="https://teacher.nwpu.edu.cn/m/en/1999000059.html" target="_blank">Yanning Zhang</a>

![framework](data/framework.png)

## Highlight
- We present a novel diagram, i.e., VadCLIP, which involves dual branch to detect video anomaly in visual classification and language-visual alignment manners, respectively. With the benefit of dual branch, VadCLIP achieves both coarse-grained and fine-grained WSVAD. To our knowledge, **VadCLIP is the first work to efficiently transfer pre-trained language-visual knowledge to WSVAD**.

- We propose three non-vital components to address new challenges led by the new diagram. LGT-Adapter is used to capture temporal dependencies from different perspectives; Two prompt mechanisms are devised to effectively adapt the frozen pre-trained model to WSVAD task; MIL-Align realizes the optimization of alignment paradigm under weak supervision, so as to preserve the pre-trained knowledge as much as possible.

- We show that strength and effectiveness of VadCLIP on two large-scale popular benchmarks, and VadCLIP achieves state-of-the-art performance, e.g., it gets unprecedented results of 84.51\% AP and 88.02\% on XD-Violence and UCF-Crime respectively, surpassing current classification based methods by a large margin.

## Training

### Setup
We extract CLIP features for UCF-Crime and XD-Violence datasets, and release these features and pretrained models as follows:

| Benchmark | CLIP[Baidu]    | CLIP | Model[Baidu]  | Model | 
|--------|----------|-----------|-------------|------------|
| UCF-Crime   | [Code: 7yzp](https://pan.baidu.com/s/1OKRIxoLcxt-7RYxWpylgLQ) | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/pengwu_stu_xidian_edu_cn/Ea86YOcp5z9KhRFDQm9a8zwBcGiGGg5BuBJtgmCVByazBQ?e=tqHLHt)     | [Code: kq5u](https://pan.baidu.com/s/1_9bTC99FklrZRnkmYMuJQw)         | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/pengwu_stu_xidian_edu_cn/Eaz6sn40RmlFmjELcNHW1IkBV7C0U5OrOaHcuLFzH2S0-Q?e=x8wtVe)           | 
| XD-Violence | [Code: v8tw](https://pan.baidu.com/s/1q8DiYHcPJtrBQiiJMI7aJw)| [OneDrive](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/Et5dWQZb2cBDs7zsrp90SrQBL_52vTRNYTdjQW6SMl0ZVA?e=foX4ph)      | [Code: apw6](https://pan.baidu.com/s/1O0uwVS3ZyDA1soWUv2VasQ) | [OneDrive](https://stuxidianeducn-my.sharepoint.com/:u:/g/personal/pengwu_stu_xidian_edu_cn/EYlNnn_xfVxBtQZuQgngrMsBHY-i8QHTVOs7PmryzQ2MyA?e=99nxnR)           | 




The CSV files contain absolute feature paths. Before training, replace the
original prefix in `list/ucf_CLIP_rgb.csv` and
`list/ucf_CLIP_rgbtest.csv` with the feature directory used by your machine or
Colab runtime.

### Paper training settings

The default arguments now follow the paper:

| Dataset | Epochs | Batch size | Learning rate | LGT window |
|---|---:|---:|---:|---:|
| UCF-Crime | 10 | 64 | `1e-5` | 8 |
| XD-Violence | 20 | 64 | `2e-5` | 64 |

Both use AdamW, temporal input length 256, CLIP ViT-B/16, context length 20,
FP32, and no gradient accumulation by default. The released UCF code applies
`batch_size=64` independently to the Normal and anomaly loaders, so one
forward contains 64 Normal plus 64 anomaly samples.

### Pooling methods

This version keeps three choices:

- `mean`: original VadCLIP Top-K arithmetic mean.
- `soft`: original K selection followed by score-based Softmax weighting.
- `multi_k`: uniform combination of Top-1%, Top-5%, Top-10%, and Top-20%
  arithmetic means.

Adaptive Instance Selection (AIS) is exposed as a separate option instead of
being mixed into `soft_topk.py`. It obtains frame anomaly probabilities from
the C-branch, computes one adaptive integer K for every paired Normal/anomaly
bag, and uses that same K for the mean Top-K pooling of both C- and A-branches.
AIS adds no learnable layer or parameter.

UCF-Crime commands:

```bash
# Original VadCLIP pooling
python src/ucf_train.py --topk-pooling mean \
  --model-path outputs/ucf_mean.pth \
  --checkpoint-path outputs/ucf_mean_checkpoint.pth \
  --log-path outputs/ucf_mean.log

# Soft Top-K
python src/ucf_train.py --topk-pooling soft \
  --c-topk-temperature 1.0 --a-topk-temperature 1.0 \
  --model-path outputs/ucf_soft.pth \
  --checkpoint-path outputs/ucf_soft_checkpoint.pth \
  --log-path outputs/ucf_soft.log

# Multi-K: Top-1%, Top-5%, Top-10%, Top-20%
python src/ucf_train.py --topk-pooling multi_k \
  --multi-k-percentages 1 5 10 20 \
  --model-path outputs/ucf_multi_k.pth \
  --checkpoint-path outputs/ucf_multi_k_checkpoint.pth \
  --log-path outputs/ucf_multi_k.log

# Adaptive Instance Selection: C-score determines a shared K for C/A
python src/ucf_train.py --topk-pooling mean \
  --adaptive-instance-selection \
  --ais-score-threshold 0.9 --ais-min-k 1 \
  --model-path outputs/ucf_ais.pth \
  --checkpoint-path outputs/ucf_ais_checkpoint.pth \
  --log-path outputs/ucf_ais.log
```

The commands inherit the paper defaults from `src/ucf_option.py`.
AIS batch logs additionally show `ais_k_mean`, `ais_k_min`, `ais_k_max`,
`ais_omega_mean`, and `ais_confident_mean`. The standalone AIS formula is in
`src/utils/adaptive_instance_selection.py`; `src/utils/soft_topk.py` remains
responsible only for the fixed mean, soft, and multi-K pooling implementations.

### Colab setup

For a complete AIS walkthrough with copy-paste Colab cells, see
[`COLAB_AIS_GUIDE.md`](COLAB_AIS_GUIDE.md).

Example when the feature directory is mounted at
`/content/drive/MyDrive/UCFClipFeatures`:

```bash
FEATURE_ROOT=/content/drive/MyDrive/UCFClipFeatures
sed -i "s#/home/xbgydx/Desktop/UCFClipFeatures#$FEATURE_ROOT#g" \
  list/ucf_CLIP_rgb.csv list/ucf_CLIP_rgbtest.csv
```

The paper used a 24 GB RTX 3090. If a Colab GPU cannot fit the physical batch
size, a memory-oriented approximation of the released code's effective batch
is:

```bash
python src/ucf_train.py --topk-pooling mean \
  --batch-size 2 --gradient-accumulation-steps 32
```

This fallback is not an exact reproduction because BatchNorm-free forward
statistics and optimizer timing can still differ from a physical batch of 64
per loader.

## References
We referenced the repos below for the code.
* [XDVioDet](https://github.com/Roc-Ng/XDVioDet)
* [DeepMIL](https://github.com/Roc-Ng/DeepMIL)

## Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@article{wu2023vadclip,
  title={Vadclip: Adapting vision-language models for weakly supervised video anomaly detection},
  author={Wu, Peng and Zhou, Xuerong and Pang, Guansong and Zhou, Lingru and Yan, Qingsen and Wang, Peng and Zhang, Yanning},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2024}
}

@article{wu2023open,
  title={Open-Vocabulary Video Anomaly Detection},
  author={Wu, Peng and Zhou, Xuerong and Pang, Guansong and Sun, Yujia and Liu, Jing and Wang, Peng and Zhang, Yanning},
  journal={arXiv preprint arXiv:2311.07042},
  year={2023}
}

```
---
