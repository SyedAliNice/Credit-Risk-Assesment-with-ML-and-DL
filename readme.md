# Customer Churn Prediction

<br>

![alt text](image.jpeg)


## Introduction

This project leverages Logistic Regression, Decision Tree, Random Forest, and Naive Bayes algorithms as well as Artificial Neural Network to predict customer churn, helping businesses proactively retain their customers.


## Dataset

The dataset consists of bank customer data with 14 features, such as age, bank balance, tenure, geography, and credit score. These features are used to predict whether a customer will exit the bank (churn). The target column indicates if the customer exited or not.
### Dataset Statistics:

Here is sample statistics of the data.

- **Name:** Churn_Modelling
- **Mode:** Tabular Data
- **Number of Samples:** 
  - Train: 12,723 
  - Test: 3,181
- **Type:** Binary Classification
- **Number of Classes:** 2
- **Classes Name:** 
  - Not Exited: 0
  - Exited: 1

## Pre-processing

The preprocessing steps of the proposed project are the following:
<list of preprocessing steps>

1. Removed unnecessary columns: Row Number, Customer ID, and Surname
2. Detected and removed outliers
3. Applied label encoding on the Age column
4. Used one-hot encoding for the Geography column
5. Balanced the target class using SMOTE to address class imbalance
6. Scaled the data using StandardScaler

## Models

For the Customer Churn Prediction following models are trained with the tuned hyperparameters:


- **Logistic Regression**
  - The Logistic Regression model was trained using default hyperparameters, with no additional tuning.

- **Decision Tree**
  - The Decision Tree model was trained using default hyperparameters, with no additional tuning.

  **Random Forest**
  - The Random Forest model was trained using default hyperparameters, with no additional tuning.

- **Naive Bayes**
  - The Naive Bayes model was trained using default hyperparameters, with no additional tuning.

### Artificial Neural Network (ANN)
- **Layers**:
  - Input Layer: 12 features (`input_dim = 12`)
  - Hidden Layer 1: 64 units, `activation = 'relu'`
  - Hidden Layer 2: 128 units, `activation = 'relu'`, BatchNormalization
  - Hidden Layer 3: 256 units, `activation = 'relu'`, BatchNormalization
  - Hidden Layer 4: 512 units, `activation = 'relu'`, BatchNormalization
  - Hidden Layer 5: 256 units, `activation = 'relu'`, BatchNormalization
  - Hidden Layer 6: 128 units, `activation = 'relu'`, BatchNormalization
  - Hidden Layer 7: 64 units, `activation = 'relu'`, BatchNormalization
  - Hidden Layer 8: 32 units, `activation = 'relu'`, BatchNormalization
  - Hidden Layer 9: 16 units, `activation = 'relu'`, BatchNormalization
  - Output Layer: 1 unit, `activation = 'sigmoid'`
  
- **Optimizer**: Adam, `learning_rate = 0.0001`
- **Loss Function**: `binary_crossentropy`
- **Metrics**: accuracy
- **EarlyStopping**: `patience = 10`, `restore_best_weights = True`
- **Epochs**: 500
- **Validation Data**: `(x_scale_test, y_test)`


## Results

| Metrics    | Logistic Regression | Decision Tree | Random Forest | Naive Bayes | ANN   |
|------------|---------------------|---------------|---------------|-------------|-------|
| Accuracy   |        0.8233       |     0.8240    |    0.8862     |   0.8004    | 0.870 |
| Precision  |        0.825        |     0.820     |    0.885      |   0.803     | 0.865 |
| Recall     |        0.825        |     0.825     |    0.890      |   0.800     | 0.865 |
| F1-Score   |        0.825        |     0.825     |    0.885      |   0.800     | 0.865 |



## Dependencies

- **NumPy version: 1.26.4**: For numerical operations.
- **Pandas version: 2.2.2**: For data manipulation and analysis.
- **Scikit-learn version: 1.4.2**: For machine learning tools.
- **Matplotlib version: 3.8.4**: For data visualization.
- **Seaborn version: 0.13.2**: For data visualization.
- **ImbLearn version: 0.12.3**: For SMOTE method for class imbalancing.
- **Tensorflow Version: 2.17.0**: For Artificial Neural Netrowk.


