# MagNet

This project consists of models of our newly-proposed MagNet, which is a multi-level attention graph network designed for accurate prediction of high-resolution HD spatial transcriptomics data. Our paper has been accepted as an oral paper by MIDL 2025!

- MagNet integrates information across multiple resolutions, including the bin, spot, and region levels, through cross-attention layers. It also extracts and combines features from neighboring regions with Graph Attention Network (GAT) and Transformer layers. 
- Furthermore, MagNet incorporates cross-resolution constraints on gene expression within the same region, further enhancing its performance in HD gene expression prediction.



![Figure2_framework](./Figure/Figure2_framework.png)

## Implementation

- To prepare the multi-resolution data format for MagNet, run files in folder `data_preprocessing`:

  ```bash
  python 1_cropped_multi_patches.py
  ...
  python 6_make_dataset.py
  ```

- To conduct 4-fold validation, run:

  ```bash
  python main.py --root_path '' --base_lr 0.005 --gpu '' --level '16' --batch_size 256
  ```

## Experiment

We tested our dataset and CRC dataset on multiple models at three resolutions and provided a list of baselines here.

![Figure_results](./Figure/Figure_results.png)

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

```
@article{zhu2025magnet,
  title={MagNet: Multi-Level Attention Graph Network for Predicting High-Resolution Spatial Transcriptomics},
  author={Zhu, Junchao and Deng, Ruining and Yao, Tianyuan and Xiong, Juming and Qu, Chongyu and Guo, Junlin and Lu, Siqi and Tang, Yucheng and Xu, Daguang and Yin, Mengmeng and others},
  journal={arXiv preprint arXiv:2502.21011},
  year={2025}
}
```
