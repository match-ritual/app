Claro, aqui está um arquivo README em inglês para o repositório GitHub de um notebook de regressão linear múltipla usando os 10 valores humanos:

```markdown
# Multiple Linear Regression with 10 Human Values

This repository contains a Jupyter Notebook that performs multiple linear regression analysis using 10 human values as predictor variables. The model aims to predict a criterion variable based on these values. The dataset used in this analysis is derived from a Google Sheets document and processed using Python libraries such as pandas and statsmodels.

## Overview

The notebook demonstrates the process of building a multiple linear regression model. It includes steps for data loading, preprocessing, model fitting, and results interpretation. The analysis provides insights into which human values significantly predict the criterion variable and explains the variance accounted for by the model.

## Features

- **Data Loading**: Loads data directly from a Google Sheets document.
- **Data Preprocessing**: Handles missing values and prepares the dataset for regression analysis.
- **Model Fitting**: Uses `statsmodels` to fit a multiple linear regression model.
- **Results Interpretation**: Analyzes the model's R-squared, coefficients, and p-values to determine significant predictors.
- **Output**: Saves the regression summary to a text file in Google Drive.

## Requirements

- Python 3.x
- pandas
- statsmodels
- google-colab (for mounting Google Drive)

## Installation

To use this notebook, you need to have the required Python libraries installed. You can install them using pip:

```bash
pip install pandas statsmodels
```

## Usage

1. **Mount Google Drive**: The notebook starts by mounting Google Drive to access the dataset and save the output.
   ```python
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)
   ```

2. **Load Data**: The dataset is loaded from a Google Sheets document using its URL.
   ```python
   sheet_url = 'https://docs.google.com/spreadsheets/d/13D2YvlGk9pkA9FVZCcvMu4jVpuoSuh7-9dWHA0r_EwE/edit?usp=sharing'
   sheet_id = sheet_url.split('/')[5]
   exported_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv'
   df = pd.read_csv(exported_url)
   ```

3. **Prepare Data**: The notebook prepares the predictor and criterion variables and handles missing data.
   ```python
   variaveis_preditivas = [col for col in df.columns if col not in ['Sexo', 'Idade', 'Filhos']]
   variavel_criterio = variaveis_preditivas[-1]
   variaveis_preditivas = variaveis_preditivas[:-1]
   df = df.dropna(subset=variaveis_preditivas + [variavel_criterio])
   ```

4. **Fit Model**: A multiple linear regression model is fitted using `statsmodels`.
   ```python
   X = df[variaveis_preditivas]
   y = df[variavel_criterio]
   X = sm.add_constant(X)
   modelo = sm.OLS(y, X).fit()
   ```

5. **Interpret Results**: The notebook interprets the model's R-squared, coefficients, and p-values.
   ```python
   r2 = modelo.rsquared
   print(f"R² (variância explicada pelo modelo): {r2:.2f}")
   significant_predictors = modelo.pvalues[modelo.pvalues < 0.05].index.tolist()
   print(f"Preditores significativos: {significant_predictors}")
   ```

6. **Save Output**: The regression summary is saved to a text file in Google Drive.
   ```python
   output_path = '/content/drive/My Drive/sumario.txt'
   with open(output_path, 'w') as f:
       f.write("Resumo da Regressão Linear Múltipla:\n")
       f.write(modelo.summary().as_text())
   ```

## Contributing

Contributions to this project are welcome. Please feel free to fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The analysis was conducted using Python and the `pandas` and `statsmodels` libraries.
- Data was sourced from a Google Sheets document and processed within a Google Colab environment.
```

This README provides a comprehensive guide for users to understand and utilize the multiple linear regression notebook. It includes an overview, features, requirements, installation instructions, usage steps, contribution guidelines, licensing information, and acknowledgments.
