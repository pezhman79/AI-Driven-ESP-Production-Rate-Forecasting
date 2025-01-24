# ESP Production Rate Prediction using Artificial Neural Networks (ANN)

This repository contains a machine learning project aimed at predicting the production rate of Electric Submersible Pumps (ESP) using an Artificial Neural Network (ANN). The project leverages the `scikit-learn` library to preprocess data, tune hyperparameters, and evaluate the model's performance.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Electric Submersible Pumps (ESPs) are widely used in the oil and gas industry to enhance production rates. Accurate prediction of ESP production rates is crucial for optimizing operations and reducing costs. This project uses an ANN to predict ESP production rates based on various input features.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy`

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/esp-production-rate-prediction.git
   cd esp-production-rate-prediction
   ```

2. **Prepare your data:**

   Place your ESP data in an Excel file named `esp.xlsx` in the `article` directory. Ensure the file has the correct format with the target variable in the 10th column (index 9).

3. **Run the script:**

   ```bash
   python esp_prediction.py
   ```

## Methodology

### Data Preparation
- The dataset is loaded from an Excel file and preprocessed.
- Features and target variables are extracted.
- The data is split into training, validation, and test sets.
- Both features and target variables are scaled using `StandardScaler`.

### Model Training
- Hyperparameter tuning is performed using `RandomizedSearchCV` to find the best parameters for the ANN.
- The ANN model is trained using the best parameters found.

### Evaluation
- The model's performance is evaluated using Mean Squared Error (MSE), Spearman's Rank Correlation Coefficient (SCC), and R-squared (R²) metrics.
- Feature importance is assessed using permutation importance.

## Results

The model's performance metrics are printed to the console, including:

- **Mean Squared Error (MSE)** for training, validation, and test data.
- **Spearman's Rank Correlation Coefficient (SCC)** for training, validation, and test data.
- **R-squared (R²)** for training, validation, and test data.

Additionally, a bar plot showing the permutation importance of each feature is displayed.


## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to reach out if you have any questions or need further assistance!
