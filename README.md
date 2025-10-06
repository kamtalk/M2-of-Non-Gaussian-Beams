M² and Modal Analysis of Non--Gaussian Beams
This repository contains the Python source code for the paper titled:
"M² and Modal Analysis of Non-Gaussian Beams: A Noise-Resilient Single-Plane Framework"
by Kenneth A. Menard
The framework presented in the paper provides a versatile and noise-tolerant tool for the complete physical characterization of a wide range of laser beam types, using only single-plane complex field measurements. This code allows for the full reproduction of all simulation results presented in the revised manuscript submitted to the Journal of Optics.
🔧 Installation
This code requires Python 3. Using a dedicated virtual environment is strongly recommended.
1. Clone the repository
code
Bash
git clone https://github.com/kamtalk/M2-of-Non-Gaussian-Beams.git
cd M2-of-Non-Gaussian-Beams
2. Create and activate a virtual environment
code
Bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
py -m venv venv
venv\Scripts\activate
3. Install required dependencies
code
Bash
pip install -r requirements.txt
🚀 Usage
The code is organized into modular scripts that can be run independently to reproduce the main results from the paper.
📊 Reproducing Table 1
Run the following script to simulate and print all beam characterization data shown in Table 1 to the console:
code
Bash
python generate_table_1.py```

### 📈 Generating the Figures

Run the following scripts to generate the figures from the revised manuscript. The output images will be saved in the project's root directory.

*   **Figure 2:** Analysis of a strongly aberrated beam with a mismatched HG basis.
    ```bash
    python generate_figure_2.py
    ```

*   **Figure 3 (New):** The framework's diagnostic capabilities, comparing a failed perturbation fit with a successful `AddModes` fit.
    ```bash
    python generate_figure_mismatch_diagnostic.py
    ```

*   **Figure 4 (Revised):** The noise robustness plot with statistically significant error bars from a Monte Carlo analysis.
    ```bash
    python Frameworks_noise_robustness.py
    ```

*   **Figure 5 (New):** A visual demonstration of the framework's noise-filtering capabilities.
    ```bash
    python generate_figure_noise_filtering.py
    ```

---

## 🧱 Code Structure

The repository is organized into executable scripts and supporting library modules.

### Main Scripts (Executable)

-   **`generate_table_1.py`**: Generates the data for Table 1.
-   **`generate_figure_2.py`**: Generates Figure 2.
-   **`generate_figure_mismatch_diagnostic.py`**: Generates the new diagnostic figure (Figure 3).
-   **`Frameworks_noise_robustness.py`**: Generates the revised noise plot (Figure 4).
-   **`generate_figure_noise_filtering.py`**: Generates the new noise-filtering figure (Figure 5).

### Supporting Modules (Libraries)

-   **`analysis_models.py`**: Contains the core implementations of the `Perturbation` and `Additive Modal Decomposition` models.
-   **`beam_definitions.py`**: Provides functions to generate all ideal, structured, and aberrated beams used in the analysis.
-   **`m2_utils.py`**: Contains utility functions for grid setup, M² calculations (both Spatial/FFT and coefficient-based), and beam profile plotting.

---

## 📚 Citation

If you use this code or the methods described in our paper for your research, please cite the original publication.

> K. A. Menard, *"M² and Modal Analysis of Non-Gaussian Beams: A Noise-Resilient Single-Plane Framework,"* Journal of Optics, [Volume, Page Numbers, Year].
>
> *(Please update with the final publication details.)*

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
