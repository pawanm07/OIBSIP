# OIBSIP
# Car Price Prediction and Sales Prediction Project

This repository contains Two distinct data analysis and prediction projects using Python and various machine learning techniques. Each project focuses on different datasets and objectives:

## 1) Car Price Prediction

### Overview

The Car Price Prediction project aims to predict the selling price of cars based on various features using machine learning algorithms.

### Dataset

Source : Car data.csv 

Description : The dataset contains information about car attributes such as year, present price, kilometers driven, fuel type, seller type, transmission type, and owner details.


### Prerequisites

- Python 3.7+
- pip (Python package installer)
- Jupyter Notebook 


### Dependencies

Install the required packages using pip:

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

### Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Key Steps

- Data preprocessing including handling missing values and categorical variables.
- Exploratory data analysis to understand relationships and distributions.
- Building and evaluating predictive models using Linear Regression and Lasso Regression.
- Visualizing model performance with scatter plots and evaluating accuracy with R-squared.

### Repository Structure
```bash
 Car_Price_Prediction.ipynb: Jupyter notebook containing the complete analysis and code.
 
 README.md: Detailed description of the project and instructions.
```



## 2) Sales Prediction

### Overview

The Sales Prediction project predicts sales based on advertising spending through TV, radio, and newspaper channels using linear regression.

### Dataset

Source: Advertising Dataset.csv

Description: The dataset includes advertising budgets for TV, radio, and newspaper, along with corresponding sales figures.


### Prerequisites

- Python 3.7+
- pip (Python package installer)
- Jupyter Notebook 


### Dependencies

Install the required packages using pip:

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install plotly
pip install scikit-learn
```

### Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn

### Key Steps

- Data loading and preprocessing including dropping unnecessary columns.
- Exploratory data analysis with correlation heatmap and scatter plots using Seaborn and Plotly.
- Building a Linear Regression model to predict sales based on advertising budgets.
- Splitting data into training and testing sets, training the model, and evaluating its performance using R-squared.

### Repository Structure

```bash
Sales_Prediction.ipynb: Jupyter notebook containing the complete analysis and code.

README.md: Detailed description of the project and instructions.
```
