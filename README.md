# VAESA: Learning A Continuous and Reconstructible Latent Space for Hardware Accelerator Design
In this work, we utilize variational autoencoders (VAEs) to enable more efficient design space exploration of deep learning accelerator designs.
For more details, please refer to:
- [ISPASS'22 VAESA Paper](https://people.eecs.berkeley.edu/~ysshao/assets/papers/vaesa-ispass2022.pdf)
- [ISPASS'22 VAESA Presentation](https://charleshong3.github.io/projects/vaesa_ispass22.pdf)
```BibTex
@inproceedings{
  huang2022vaesa,
  title={Learning A Continuous and Reconstructible Latent Space for Hardware Accelerator Design},
  author={Qijing Huang and Charles Hong and John Wawrzynek and Mahesh Subedar and Yakun Sophia Shao},
  booktitle={International Symposium on Performance Analysis of Systems and Software (ISPASS)},
  year={2022},
  url={https://people.eecs.berkeley.edu/~ysshao/assets/papers/vaesa-ispass2022.pdf}
}
```

## Installation

1. Install python dependencies
```
python -m pip install -r requirements.txt
```

2. CoSA (includes Gurobi, Timeloop, Accelergy):
Please refer to the instructions in the [CoSA repo](https://github.com/ucb-bar/cosa) to install CoSA and its dependencies. Define the `COSA_DIR` environment variable pointing to the directory where CoSA is installed.
```
export COSA_DIR=<path/to/cosa/dir>
```

## Train VAESA

To train the default configuration of VAESA, simply run: `bash run.sh` from the `src` directory.

The file `src/run.sh` contains the following variables which can configure the models trained:

```
SEED         - random seed (default: 1234)
NZ           - latent space dimensionality (default: 4)
EPOCHS       - number of epochs to train (default: 2000)
DATASET_SIZE - number of training data points to use (default: 131328)
PRED_MODEL   - set predictor model. options [orig, deep, orig_1, deep_1] (default: orig_1)
VAE_MODEL    - set VAE hidden_dims model. options [orig, model_1, model_2] (default: model_1)
DATASET_PATH - training data path (default: ../db/dataset_all_layer.csv)
OBJ          - optimization target. options [edp, latency, energy] (default: edp)
```
