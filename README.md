# Heart Disease Prediction

## Overview
This project aims to predict the presence of heart disease in patients using machine learning algorithms. The prediction is based on various medical attributes and patient data. The goal is to assist medical professionals in identifying at-risk patients and making informed decisions.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Evaluation](#models-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset
The dataset used in this project is the [Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) from the UCI Machine Learning Repository. It contains 303 instances and 14 attributes, including:
- Age
- Sex
- Chest pain type (4 values)
- Resting blood pressure
- Serum cholestoral in mg/dl
- Fasting blood sugar > 120 mg/dl
- Resting electrocardiographic results (values 0, 1, 2)
- Maximum heart rate achieved
- Exercise induced angina
- Oldpeak = ST depression induced by exercise relative to rest
- The slope of the peak exercise ST segment
- Number of major vessels (0-3) colored by flourosopy
- Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/aritraghosh7/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Load the dataset and preprocess it:
   ```python
   import pandas as pd

   data = pd.read_csv('heart_disease.csv')
   # Preprocessing steps here
   ```

2. Train the machine learning model:
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier

   X = data.drop('target', axis=1)
   y = data['target']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   ```

3. Evaluate the model:
   ```python
   from sklearn.metrics import accuracy_score

   y_pred = model.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   print(f'Accuracy: {accuracy}')
   ```

## Models and Evaluation
This project explores several machine learning algorithms, including:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

The models are evaluated based on metrics such as accuracy, precision, recall, and F1-score.

## Results
The Random Forest classifier achieved the highest accuracy of 85%. Detailed results and performance metrics are available in the `results` folder.

## Contributing
Contributions are welcome! If you have any suggestions or improvements, please create an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
