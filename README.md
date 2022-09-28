# DirecFormer: A Directed Attention in Transformer Approach to Robust Action Recognition

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
