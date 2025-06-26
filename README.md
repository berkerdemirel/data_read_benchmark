# Data Read Benchmark

This project benchmarks data reading performance for different file formats in PyTorch. It compares reading speeds for PNG, Arrow, Parquet, and NPY formats, using various batch sizes and numbers of workers. The project uses PyTorch Lightning for streamlined training and Weights & Biases for logging and visualizing results.

## Features

- **Data Generation**: Creates a dummy dataset with specified parameters (number of classes, samples, image size).
- **Multiple File Formats**: Supports benchmarking for PNG, Arrow, Parquet, and NPY.
- **Configurable**: Easily configure data, dataloader, and trainer parameters using a `config.yaml` file.
- **Benchmarking**:
    - Compares data loading performance with different batch sizes and numbers of workers.
    - Supports both single-GPU and multi-GPU setups.
- **Logging**: Logs results to Weights & Biases for easy comparison and analysis.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/data-read-benchmark.git
   cd data-read-benchmark
   ```

2. **Set up the environment and install dependencies:**

   This project uses `uv` for fast dependency management. First, install `uv`:
   ```bash
   pip install uv
   ```
   Next, create a virtual environment and activate it:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
   Finally, install the project dependencies from `pyproject.toml`:
   ```bash
   uv pip install -e .
   ```

## Usage

### Data Generation

If the dataset is not already generated, the script will automatically create it based on the parameters in `config.yaml`. You can customize the data generation process by modifying the `data` section of the config file.

### Running the Benchmark

To run the benchmark, execute the `main.py` script:

```bash
python main.py
```

The script will iterate through all combinations of data formats, batch sizes, and worker numbers specified in `config.yaml`.

## Configuration

The `config.yaml` file allows you to configure the benchmark parameters:

- **`seed`**: Random seed for reproducibility.
- **`data`**:
    - **`num_classes`**: Number of classes in the dataset.
    - **`num_samples`**: Total number of samples to generate.
    - **`image_size`**: Dimensions of the images.
    - **`formats`**: List of file formats to benchmark.
    - **`root_dir`**: Directory to store the generated data.
    - **`parquet_batch_size`**: Batch size for writing Parquet files.
- **`dataloader`**:
    - **`batch_size`**: List of batch sizes to test.
    - **`num_workers`**: List of worker numbers to test.
- **`trainer`**:
    - **`gpus_configs`**: GPU configurations for training (e.g., `[0]` for single-GPU, `[0, 1, 2, 3]` for multi-GPU).
    - **`max_epochs`**: Number of epochs for the training loop.
- **`wandb`**:
    - **`project`**: Name of the Weights & Biases project.
    - **`entity`**: Your Weights & Biases entity.


## Dependencies

- [PyTorch](https://pytorch.org)  
- [PyTorch Lightning](https://www.pytorchlightning.ai)  
- [Hydra](https://hydra.cc)  
- [Weights & Biases](https://wandb.ai)  
- [NumPy](https://numpy.org)  
- [Pandas](https://pandas.pydata.org)  
- [PyArrow](https://arrow.apache.org/docs/python)  
- [Scikit-learn](https://scikit-learn.org)  
