# VexelNet

Official implementation for paper [Vexel-Net: Vectors-Composed Features Oriented Neural Network](www.bing.com) .

---

## Abatract

Extracting features that are robust to transformation and perturbation is important for all deep learning tasks and related applied areas. In this research, we dedicate to designing a method that can extract more robust features for resisting perturbation and attack upon input data, with operators bearing more sophisticated transformations. For this purpose, we treat features as composite of vectors, namely, **Vexels**, for letting features experience more affine computation. And we introduce an **A and B Mixed Weights** algorithm as well as four operators for handling vexels with more affine computations. Above all, we build our VexelNet by reshapeing features from CNNs, into Vexels, then feed them to our algorithm and operators. Through experiments, we demonstrate effectiveness and advantages of our approach to compared methods, on experiments of test on data under adversarial attacks and affine transformations. Eventually, we improve some drawbacks of CNNs and CapsNets, like, massive parameters and calculations, vulnerable to adversarial attacks, sensitive to perturbation on input data etc. .



## Acknowledgement

Our codes are implemented benefiting from [SR-CapsNet](https://github.com/coder3000/SR-CapsNet),  we recommend that interested readers will gain a lot after reading their paper [Self-Routing Capsule Networks](https://papers.nips.cc/paper_files/paper/2019/hash/e46bc064f8e92ac2c404b9871b2a4ef2-Abstract.html) and looking into their repository.



## Requirement

```python
pytorch >= 2.0.1+cu118
pytorch lightning >= 2.1.0
pytorch vision >= 0.15.2+cu118
einops >= 0.6.1
numpy >= 1.23.5
```

Other python libraries need for this repository will be installed when installing pytorch lightning. If  met problem with libraries for this repository, you are welcome to submit issues to us.



### About args

Args are defined in [utils/configUtils.py](utils/configUtils.py) , if users want to specify different values for args, we suggest modifying args in [utils/configUtils.py](utils/configUtils.py), just cleanly run [main.py](main.py).  And other arguments for models and optimization are defined in [main.py](main.py) as python dictoinaries.



### Image Classification

The configurations and parameters of networks for image classifications are already recorded in [our paper](www.bing.com), please refer to it if interested.



### Affine Robustness

Just  modify args for viewpoints by assign values for `--exp` and ` famaliar`,  look at [data/data.py](data/data.py), [data/utils.py](data/utils.py) for how these args work.



### Adversarial Attack

Specify values for args `--typeAttack`, `--epsAttack` and `--targeted` in [utils/configUtils/py](utils/configUtils/py) in order to generate adversarial samples and run [adversarial_attack.py](adversarial_attack.py) for adversarial attack test. Interested readers can refer to [adversarial_attack.py](adversarial_attack.py)  for loading checkpoints for `LightningModule`.



### Ablation

All ablation experiments are conducted by changing `args` in main.py and changing network architecture in [models/networks/VexNet.py](models/networks/VexNet.py). Please pay attention to `channels`, `kernel_size`, and `padding` of Vexel `conv` layers when modifying network architecture.  We listed used configurations as follows.

#### Vex-1

|          | kernel_size        | stride | padding | in_channels     | out_channels |
|:-------- | ------------------ |:------:|:-------:|:---------------:|:------------:|
| VexConv1 | input spatial size | 1      | 0       | initVexChannels | numClasses   |

#### Vex-2

|          | kernel_size        | stride | padding | in_channels         | out_channels       |
|:-------- | ------------------ |:------:|:-------:|:-------------------:|:------------------:|
| VexConv1 | 5                  | 2      | 2       | initVexChannels     | initVexChannels//2 |
| VexConv2 | input spatial size | 1      | 0       | initVexChannels //2 | numClasses         |

As for ablation on initial Vexel **channels** and **dimension**, just refer to [main.py](main.py) and modify corresponding values in `argsNet`.



### Logs and Checkpoints

We adopt [Pytorch](https://pytorch.org/get-started/locally/) and [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) for implementation our methods as well as other networks, please refer to [doc](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#id3)  of `load_from_checkpoint` for loading trained `LightningMotule` checkpoints.
