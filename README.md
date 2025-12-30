# petrol-consumption-prediction
This project implements a Random Forest Regressor to predict the petrol consumption of a given region based on several key features like tax rates, income levels, and infrastructure.

**📌 Project Overview**
The goal of this project is to analyze how various factors influence petrol usage. The model is trained on a dataset containing historical data of petrol consumption across different regions and uses the Scikit-Learn library to make predictions.

**📊 Dataset Features**
- The model utilizes the following features for prediction:
- Petrol_tax: The tax rate on petrol in the region.
- Average_income: The average annual income of the population.
- Paved_Highways: The total length of paved highways (in miles).
- Population_Driver_licence(%): The percentage of the population that holds a valid driver's license.
- Petrol_Consumption (Target): The amount of petrol consumed (gallons).

**🛠️ Requirements**
- Python 3.x
- Pandas
- Scikit-Learn
  
**🚀 Implementation Steps**
- Data Loading: Importing the dataset using Pandas.
- Feature Selection: Defining independent variables (X) and the target variable (y).
- Data Splitting: Splitting the data into 80% training and 20% testing sets.
- Model Training: Initializing and training a RandomForestRegressor with 200 estimators.
- Evaluation: Measuring performance using R² Score and Mean Absolute Error (MAE).
- Prediction: Testing the model with custom input values to predict consumption.

**📈 Results**
The current model evaluation on the test set yielded:
- Mean Absolute Error: ~52.04
- R² Score: ~0.043
- **Note:** The low R² score suggests that the current feature set or dataset size may need further optimization (like feature engineering or hyperparameter tuning) to improve predictive accuracy.
