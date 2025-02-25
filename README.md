# DyEdgeGAT: Dynamic Edge via Graph Attention for Early Fault Detection in IIoT Systems
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dynamic-graph-attention-for-anomaly-detection/unsupervised-anomaly-detection-on-pronto)](https://paperswithcode.com/sota/unsupervised-anomaly-detection-on-pronto?p=dynamic-graph-attention-for-anomaly-detection)

Our paper has been officially accepted for publication in the **IEEE Internet of Things Journal**, and is now available online. You can access it via the following DOI link: [DOI: 10.1109/JIOT.2024.3381002](https://doi.org/10.1109/JIOT.2024.3381002).

## Data
1. The synthetic dataset is included in the `run/datasets/toy` folder.
2. To prepare the Pronto dataset, download the raw data from [PRONTO heterogeneous benchmark dataset](https://zenodo.org/records/1341583) and put it in the `run/datasets/pronto/raw` folder. Then, run the execute `run/datasets/pronto/train_test_split.ipynb` to prepare the dataset.

## Code Structure
The code is organized in the following way:
- `.vscode/` contains the configuration files for debugging in visual studio code
- `src/model/` contains the implementation of the models
- `src/utils/` contains the implementation of the utility functions
- `src/train/` contains the implementation of the training functions
- `run/data/` contains the datasets used in the experiments
- `run/configs/` contains the configuration files used to run the experiments
- `run/main.py` is the main file used to run the experiments
- `run/evaluate.py` is the main file used to evaluate the models

## Environment Setup
This project relies on specific dependencies and packages, which are defined in the eff_env.yml file. You can set up the environment using Conda by running the following command:

```bash
conda env create -f env.yml
```

If you want to update the environment, you can run the following command:

```bash 
conda env update --file env.yml  --prune
```

### Installing torch with GPU support
To install PyTorch with CUDA support, use the following command:

```bash 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

Depending on your CUDA version, you may need to change the `cudatoolkit` version.
Detailed instructions can be found [here](https://pytorch.org/get-started/previous-versions/).
Note pyTorch 1.12.* binaries do not support CUDA versio above (including) 11.7.


### Installing pytorch-geometric
Follow [PyG 2.2.0 INSTALLATION Guide](https://pytorch-geometric.readthedocs.io/en/2.2.0/notes/installation.html) for detailed instructions.


## Usage
To train the model, run the following command:

```bash 
python run/main.py --cfg run/configs/toy/dyedgegat.yaml --repeat 5
```

The `--cfg` argument specifies the path to the config file, and the `--repeat` argument specifies the number of times to repeat the experiment.

For evaluation, run the following command:

```bash
python run/evaluate.py --cfg run/configs/toy/dyedgegat.yaml 
```

For any questions or feedback, please open an issue in this repository or contact us directly via email.
