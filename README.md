# Deep Generative Classification of Blood Cell Morphology

This repository contains the code accompanying the paper ["Deep Generative Classification of Blood Cell Morphology"](https://arxiv.org/abs/2408.08982), which is published as a preprint on arXiv and is currently under peer review. The code demonstrates the application of diffusion-based models for classification tasks, with a focus on blood cell morphology. It provides a foundation for reproducing key findings and offers a framework for further exploration in this area.

## Key Features

* **Generative Classification:** Implements a diffusion-based classifier for robust and accurate blood cell classification.
* **Anomaly Detection:** Demonstrates superior anomaly detection capabilities compared to traditional discriminative models.
* **Uncertainty Quantification:** Provides reliable uncertainty estimates, allowing for better assessment of model predictions.
* **Domain Shift Robustness:** Exhibits resilience to variations in imaging conditions, enhancing generalisability.
* **Data Efficiency:** Achieves high performance even in low-data regimes, a crucial advantage in medical imaging.
* **Explainability:** Generates counterfactual heatmaps, providing interpretable insights into model decisions.

## Getting Started

### Prerequisites

* **GPU:**  $\ge$ 24GB RAM (for smaller GPUs, decrease the batch size in `EXAMPLE.sh`)
* **Operating System:** Tested on Ubuntu 20.04.6 and Ubuntu 22.04.4
* **CUDA:** Tested on CUDA 11.8 and 12.5

### Installation

1. **Clone the repository:**

   ```
   git clone git@github.com:Deltadahl/CytoDiffusion.git
   cd CytoDiffusion
   ```

2. **Create and activate the conda environment:**

   ```
   conda env create -f environment.yml
   conda activate CytoDiffusion
   ```

3. **Configure Accelerate:**
   Run the following command:
   ```
   accelerate config
   ```
   When prompted, provide these answers for a simple single GPU setup:
   - Compute environment: This machine
   - Machine type: No distributed training
   - Run training on CPU only: NO
   - Optimize script with torch dynamo: NO
   - Use DeepSpeed: NO
   - GPU(s) to use: 0
   - Enable numa efficiency: NO
   - Use mixed precision: fp16

4. **Log in to Weights & Biases (wandb):**
   ```
   wandb login
   ```
   Follow the prompts to complete the login process.

### Running the Example Code

1. **Prepare the example data:**

    ```
   cd data/prepare_data
   python prepare_data.py
    ```
   Provide the path to `example_data` (located in the current folder) when prompted.

2. **Train the model:**

   ```
   cd ../../train_and_test
   sh EXAMPLE.sh
   ```

## Using Your Own Dataset

To use your own dataset, provide the path to your dataset when you run `prepare_data.py`
For example:

```plaintext
your_dataset
├── basophil
│   ├── image1.png
│   └── image2.png
├── eosinophil
│   ├── image3.png
│   └── image4.png
├── ...
└── name_to_number.json
```
Then, update the paths in the `EXAMPLE.sh` script accordingly.

## Configuration and Reproducibility

We provide several options for configuring and running experiments:

1. **Basic Configuration:** For initial setup and testing, we recommend using the `EXAMPLE.sh` script located in the `train_and_test` folder. This script serves as a template for setting essential parameters such as data paths, training steps, and other relevant settings.

2. **Reproducing Experiments:** To facilitate the reproduction of our experimental results, we have included additional `.sh` scripts in the same folder as `EXAMPLE.sh`. These scripts contain the specific configurations used in our experiments.

3. **Custom Experiments:** Feel free to create your own `.sh` scripts based on our examples to explore different configurations and scenarios.

### Running Experiments

To run any of these scripts, follow these steps:

1. **Prepare the Data:**
   - For `EXAMPLE.sh`, follow the data preparation steps in the "Getting Started" section.
   - For other experiment scripts or custom datasets:
     a. Navigate to the data preparation folder:
        ```
        cd data/prepare_data
        ```
     b. Run the data preparation script:
        ```
        python prepare_data.py
        ```
     c. When prompted, provide the path to your dataset.

2. **Update Script Paths:**
   - Open the `.sh` script you want to use.
   - Update the data paths in the script to match your prepared dataset location.

3. **Run the Script:**
   - Navigate to the `train_and_test` folder:
     ```
     cd ../../train_and_test
     ```
   - Execute the desired script:
     ```
     sh EXAMPLE.sh
     # or
     sh <sh_name>.sh
     ```

## Datasets

The code is tested on the following datasets:

* **PBC Dataset:**  Acevedo et al. [A dataset of microscopic peripheral blood cell images for development of automatic recognition systems](https://www.sciencedirect.com/science/article/pii/S2352340920303681)
* **Raabin-WBC Dataset:** Kouzehkanan et al. [A large dataset of white blood cells containing cell locations and types, along with segmented nuclei and cytoplasm](https://www.nature.com/articles/s41598-021-04426-x)
* **Bodzas Dataset:** Bodzas et al. [A high-resolution large-scale dataset of pathological and normal white blood cells](https://europepmc.org/article/MED/37468490)
* **Our Custom Dataset:** See [our paper](https://arxiv.org/abs/2408.08982)

## Expected Performance

The model achieves an accuracy of >80% when you run the example dataset. The `EXAMPLE.sh` script will save the trained model locally and log training information to Weights & Biases.

## Contact
For questions or collaboration opportunities, please contact:\
Simon Deltadahl: scfc3@cam.ac.uk

## Reporting Issues

Please report any issues or bugs on the [Issues page](https://github.com/Deltadahl/CytoDiffusion/issues).

## Licence

This code is licenced under the Apache 2.0 Licence.

## Citation

If you use this code in your research, please cite our paper:

```
@article{deltadahl2024deep,
  title={Deep Generative Classification of Blood Cell Morphology},
  author={Deltadahl, Simon and Gilbey, Julian and Van Laer, Christine and Boeckx, Nancy and Leers, Mathie and Freeman, Tanya and Aiken, Laura and Farren, Timothy and Smith, Matt and Zeina, Mohamad and {BloodCounts! consortium} and Rudd, James HF and Piazzese, Concetta and Taylor, Joseph and Gleadall, Nicholas and Schönlieb, Carola-Bibiane and Sivapalaratnam, Suthesh and Roberts, Michael and Nachev, Parashkev},
  journal={arXiv preprint arXiv:2408.08982},
  year={2024}
}
```
