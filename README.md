# Optimizing Bio-Inspired Deep Convolutional Neural Networks

**Author:** Luis Morales Layja  
**Master's Thesis Project**

## Introduction

This repository contains the code and documentation for the research project regarding the optimization of Bio-Inspired Deep Convolutional Neural Networks.

This project is built upon the **BioNet** framework developed by [Evans et al. (2022)](https://github.com/bdevans/BioNet). 

### Key Contributions
The main goal of this project was to extend the VGG-16 architecture with Gabor filters by implementing bio-inspired mechanisms:

1.  **Retinal Information Bottleneck (RIB):** Simulating the compression of visual information found in the retina.
2.  **Recurrent Connections (RCs):** Incorporating ConvLSTM layers in the first convolutional block to mimic feedback loops in the visual cortex.

The introduced properties were a bottleneck, mimicking the reduction in ganglion cells as visual information passes from the retina through the optic nerve in primates, and recurrent connections, which are prevalent in the visual cortex. 

These additions aim to improve the model’s generalization capacity and robustness to noise by creating an architecture more similar to the visual cortex, thereby narrowing the performance gap between humans and models.

---

## Directory Structure

The expected directory structure before training is as follows:

```text
.
├── bionet
│   ├── config.py       # Constants and configuration parameters
│   ├── explain.py
│   ├── __init__.py
│   ├── plots.py
│   ├── assess.py       # Generates CSV metrics from results
│   ├── bases.py        # CNN Architectures (Includes RIB and RCs implementations)
│   ├── utils.py        # Auxiliary functions and custom filters (Gabor)
│   └── preparation.py  # Data preprocessing and noise generation
├── data
│   └── CIFAR-10G       # Dataset folder
├── results             # Generated after training
├── model.py            # Main execution script
└── README.md
```

## Installation & Setup

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/layja13/Bio-Inspired-DCNNs.git
    ```

2.  **Download the Dataset:**
    Clone the [CIFAR-10G](https://github.com/bdevans/CIFAR-10G) generalisation test set into the `data/` directory.

---

## Usage and Execution

The main entry point is `model.py`. This script handles training, testing, and the incorporation of the new bio-inspired modules.

### New Arguments
In addition to the original BioNet arguments, this repository introduces:

*   `--bottleneck [INT]`: Defines the size of the Retinal Information Bottleneck (e.g., `2`).
*   `--recurrent`: Boolean flag. If present, adds Recurrent Connections (LSTM) to the model.
*   `--continue_train`: Allows resuming training from saved weights.


### Examples

Below are the commands to reproduce the experiments conducted in this research.

#### 1. Baseline Model (VGG-16 + Gabor Filters)
Training and evaluation of the standard bio-inspired model introduced by [Evans et al. (2022)] without new additions.

```bash
python model.py --convolution Gabor --base VGG16 --epochs 100 \
--train --use_initializer --clean --use_initializer \
--data_augmentation --test_generalisation \
--recalculate_statistics --test_perturbations --save_predictions \
--log --verbose 1 --gpu 0 --label Bottleneck --batch 32 \
--seed 42
```

#### 2. Retinal Information Bottleneck (RIB) Incorporation
Incorporates a bottleneck of size 2.

```bash
python model.py --convolution Gabor --base VGG16 --epochs 100 \
--train --use_initializer --clean --bottleneck 2 \
--use_initializer --data_augmentation --test_generalisation \
--test_perturbations --save_predictions --recalculate_statistics \
--log --verbose 1 --gpu 0 --label Bottleneck --batch 32 \
--seed 42
```

#### 3. Recurrent Connections (RCs) Incorporation
Incorporates Recurrent Connections (ConvLSTM) into the first layer.

```bash
python model.py --convolution Gabor --base VGG16 --epochs 100 \
--train --clean --recurrent --use_initializer \
--data_augmentation --test_generalisation --test_perturbations \
--save_predictions --recalculate_statistics --log --verbose 1 \
--gpu 0 --label Recurrent --batch 32 --seed 42
```

#### 4. Combined Model (RIB + RCs)
Incorporates both the bottleneck and recurrent connections.

```bash
python model.py --convolution Gabor --base VGG16 --epochs 100 \
--train --clean --recurrent --bottleneck 2 --use_initializer \
--data_augmentation --test_generalisation --test_perturbations \
--save_predictions --recalculate_statistics --log --verbose 1 \
--gpu 0 --label Recurrent --batch 32 --seed 42
```

---

## Results

After execution, the script generates a `results/` folder containing:
*   **Metrics:** CSV files with accuracy on training sets and perturbed versions.
*   **Predictions:** Category predictions for analysis.

You can use the `assess.py` script to process these results into consolidated CSV files for plotting.

---

## References

1.  **Evans, B. D., Malhotra, G., & Bowers, J. S. (2022).** Biological convolutions improve DNN robustness to noise and generalisation. *Neural Networks*, 148, 96–110. [DOI: 10.1016/j.neunet.2021.12.005](https://doi.org/10.1016/j.neunet.2021.12.005)
```