# Taxi Fare Prediction

This project builds and evaluates a machine learning pipeline to predict New York City taxi fares.  
It combines data preprocessing, feature engineering, exploratory analysis, clustering, and a deep learning model built with TensorFlow/Keras.

---

## Project Overview
The goal is to predict the **fare amount** of NYC taxi rides given trip details such as pickup/dropoff coordinates, datetime, and passenger count.  
The workflow includes:
- Data cleaning and filtering of invalid/missing values.
- Geospatial processing using the **Haversine distance formula**.
- Exploratory Data Analysis (EDA) with visualizations.
- Geographical clustering with **K-Means**.
- Time-based feature extraction (hour, day of week, month).
- Deep learning regression model with TensorFlow.

---

## Requirements
Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow pillow requests
```

### Libraries Used
- `pandas`, `numpy` – data handling
- `matplotlib`, `seaborn` – visualization
- `scikit-learn` – preprocessing, clustering, evaluation metrics
- `tensorflow` – deep learning regression model
- `Pillow`, `requests` – map image loading

---

## Dataset
The dataset comes from the **New York City Taxi Fare Prediction** competition on Kaggle.

- File: `train.csv`  
- A subset of 1,000,000 rows is used for processing.
- Key columns:
  - `pickup_longitude`, `pickup_latitude`
  - `dropoff_longitude`, `dropoff_latitude`
  - `fare_amount`
  - `pickup_datetime`

---

## Feature Engineering
1. **Filtering invalid coordinates** (bounding box around NYC).  
2. **Haversine distance** computation between pickup and dropoff.  
3. **Removal of unrealistic fares** (fares < 2 or extremely large values).  
4. **Clustering** pickup/dropoff points into 20 clusters.  
5. **Datetime features** extracted: hour, weekday, and month.

---

## Exploratory Data Analysis
- Scatterplots of pickup/dropoff locations overlaid on a NYC map.
- Heatmaps of trip density.
- Fare amount relationships with:
  - Distance
  - Time of day
  - Day of week
  - Month
  - Pickup/Dropoff clusters

---

## Model
A deep learning regression model using TensorFlow/Keras:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(n_features,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

- **Loss:** Mean Absolute Error (MAE)  
- **Optimizer:** Adam  
- **Callbacks:** EarlyStopping, ReduceLROnPlateau  
- **Training:** 10 epochs, batch size = 256  

---

## Results
Model performance on the test set:
- **Mean Absolute Error (MAE):** ~1.88  
- **Root Mean Squared Error (RMSE):** ~4.25  

Error distribution is approximately centered near 0, showing reasonable prediction accuracy.

---

## How to Run
1. Clone the repo / copy the notebook.  
2. Download the dataset from Kaggle and place it in the project directory.  
3. Run the preprocessing, visualization, and model training cells sequentially.  
4. Evaluate model performance on the held-out test set.

---

## Visualizations
The notebook generates:
- Pickup and dropoff maps over NYC.
- Heatmaps of location densities.
- Boxplots of fare by time and cluster.
- Model training history curves.
- Prediction error histograms.

---

## Future Improvements
- Experiment with gradient boosting models (XGBoost, LightGBM).  
- Hyperparameter tuning for neural network layers/neurons.  
- Feature engineering with weather, traffic, or holiday data.  
- Deploy as an API for real-time fare predictions.  
