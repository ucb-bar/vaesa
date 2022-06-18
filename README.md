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

TODO: Update CoSA installation instructions below

1. Obtain a Gurobi license (see [here](https://www.gurobi.com/academia/academic-program-and-licenses/) for instructions on obtaining one for free if you're an academic). You do **not** need to download or install Gurobi itself. Once you have a license, download and extract the [Gurobi license manager](https://support.gurobi.com/hc/en-us/articles/360059842732-How-do-I-set-up-a-license-without-installing-the-full-Gurobi-package-), then run the `grbgetkey` executable, supplying your [license key](https://www.gurobi.com/downloads/licenses/) when required. If you select a non-default location for the license file, specify the location of the file using:
```
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```
2. Timeloop (optional - can be skipped if you only want to run the scheduler, without Timeloop benchmarking): 
Please refer to the instructions in the [Timeloop Tutorial](http://accelergy.mit.edu/infra_instructions.html) to install Timeloop with Docker.
To install from source code please, follow the instructions in [Timeloop Github](https://github.com/NVlabs/timeloop).
The specific Timeloop version used for CoSA evaluation is commit [11920be](https://github.com/NVlabs/timeloop/commit/11920be5a744239c985ff049256f2fc40f65ce8b). 

## Train VAESA

To train the default configuration of VAESA, simply run: `bash src/run.sh` from the command line.

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

## Inference

TODO
