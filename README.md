# DirecFormer: A Directed Attention in Transformer Approach to Robust Action Recognition

This is the official implementation of the paper "[DirecFormer: A Directed Attention in Transformer Approach to Robust Action Recognition](https://openaccess.thecvf.com/content/CVPR2022/papers/Truong_DirecFormer_A_Directed_Attention_in_Transformer_Approach_to_Robust_Action_CVPR_2022_paper.pdf)".

[![DirecFormer](https://img.youtube.com/vi/7IGyh0oWaQs/0.jpg)](https://www.youtube.com/watch?v=7IGyh0oWaQs)


## Installation

The installation of this repository is similar to the installation of SlowFast. The instruction can be found [here](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md)

To prepare a dataset, you should follow the instructions [here](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/DATASET.md) provided by SlowFast.

## Testing

To test the model on the Jester dataset, you can perform the following commands:

```bash
python tools/run_net_tsm.py --cfg config/Jester/DirecFormer.yaml \
            TRAIN.ENABLE False \
            TEST.CHECKPOINT_FILE_PATH <PATH-TO-CHECKPOINT> \
```

## Training and Optimization

Please contact the Project Investigator (Khoa Luu) for further information about training models, optimized models on-the-edge and low-cost devices.


## Acknowledgements

This codebase is borrowed from [SlowFast](https://github.com/facebookresearch/SlowFast) and [TimeSFormer](https://github.com/facebookresearch/TimeSformer)


## Citation
If you find this code useful for your research, please consider citing:
```
@inproceedings{truong2021direcformer,
  title={DirecFormer: A Directed Attention in Transformer Approach to Robust Action Recognition},
  author={Truong, Thanh-Dat and Bui, Quoc-Huy and Duong, Chi Nhan and Seo, Han-Seok and Phung, Son Lam and Li, Xin and Luu, Khoa},
  booktitle={Computer Vision and Pattern Recognition},
  year={2022}
}
```
