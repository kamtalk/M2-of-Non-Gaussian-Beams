# M² and Modal Analysis of Non-Gaussian Beams

This repository contains the Python source code for the paper titled:

**"M² and Modal Analysis of Non-Gaussian Beams: A Noise-Resilient Single-Plane Framework"**  
by Kenneth A. Menard

The framework presented in the paper provides a versatile and noise-tolerant tool for the complete physical characterization of a wide range of laser beam types, using only single-plane complex field measurements. This code allows for full reproduction of all simulation results presented in the manuscript.

---

## 🔧 Installation

This code requires Python 3. It is strongly recommended to use a dedicated virtual environment.

### 1. Clone the repository

```bash
git clone https://github.com/kamtalk/M2-of-Non-Gaussian-Beams.git
cd M2-of-Non-Gaussian-Beams
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install required dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

The code is organized into modular scripts to reproduce the main results from the paper.

### 📊 Reproducing Table 1: Beam Characterization Results

Run the following script to simulate and print all beam characterization data shown in Table 1:

```bash
python generate_table_1.py
```

This script analyzes each beam type and prints the M² and modal decomposition results to the console.

### 📈 Generating Figure 2: Aberrated Beam Analysis

To generate the multi-panel figure analyzing a strongly aberrated TEM₀₀ beam:

```bash
python generate_figure_2.py
```

The image file will be generated and saved in the current directory.

### 📉 Generating Figure 3: Noise Robustness Comparison

To generate the figure comparing the M² estimation methods under increasing noise:

```bash
python Frameworks_noise_robustness.py
```

This will create and save the noise robustness plot in the current directory.

---

## 🧱 Code Structure

The repository is organized into executable scripts and supporting modules.

### Main Scripts

- **`generate_table_1.py`**  
  Simulates and summarizes M² and modal content for all beam types used in Table 1.

- **`generate_figure_2.py`**  
  Generates the multi-panel visualization for the strongly aberrated beam case (Figure 2).

- **`Frameworks_noise_robustness.py`**  
  Produces the plot comparing noise robustness of different M² methods (Figure 3).

### Supporting Modules

- **`analysis_models.py`**  
  Contains the implementation of the Perturbation and Additive Modal Decomposition frameworks.

- **`beam_definitions.py`**  
  Provides functions to generate all structured, non-Gaussian, and aberrated beams used in the analysis.

- **`m2_utils.py`**  
  Utility functions for calculating M² using both the spatial/FFT-based method and the coefficient-based method.

---

## 📚 Citation

If you use this code or the methods described in our paper for your research, please cite the original publication:

> K. A. Menard, *"M² and Modal Analysis of Non-Gaussian Beams: A Noise-Resilient Single-Plane Framework,"* Journal of Optics, [Volume, Page Numbers, Year].  
> *(Please update the citation once the paper is published.)*

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
