# Bar Chart - Anomaly Scores
import lightningchart as lc
import pandas as pd
from scipy.stats import chi2
from datetime import datetime
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import changefinder
import numpy as np

lc.set_license('my-license-key')

file_path = 'machine_temperature_system_failure.csv'
data = pd.read_csv(file_path)

data['timestamp'] = pd.to_datetime(data['timestamp'])

def calculate_anomaly_scores(data):
    # Hotelling's T²
    mean = data['value'].mean()
    std = data['value'].std()
    data['hotelling_score'] = ((data['value'] - mean) / std) ** 2

    # One-Class SVM
    ocsvm_model = OneClassSVM(nu=0.2, gamma=0.001, kernel='rbf')
    ocsvm_model.fit(data[['value']])
    data['ocsvm_score'] = ocsvm_model.decision_function(data[['value']])

    # Isolation Forest
    iforest_model = IsolationForest(n_estimators=300, contamination=0.1, max_samples=700, random_state=42)
    iforest_model.fit(data[['value']])
    data['iforest_score'] = iforest_model.decision_function(data[['value']])

    # LOF (Local Outlier Factor)
    lof_model = LocalOutlierFactor(n_neighbors=500, contamination=0.07)
    lof_scores = lof_model.fit_predict(data[['value']])
    data['lof_score'] = lof_model.negative_outlier_factor_

    # ChangeFinder
    cf_model = changefinder.ChangeFinder(r=0.002, order=1, smooth=250)
    data['cf_score'] = [cf_model.update(value) for value in data['value']]
    
    return data

data = calculate_anomaly_scores(data)

def plot_histogram(data, column, title, chart):
    counts, bins = np.histogram(data[column], bins=50)
    bar_data = [{'category': str(bins[i]), 'value': int(counts[i])} for i in range(len(counts))]

    chart.set_data(bar_data)
    chart.set_title(title)

dashboard = lc.Dashboard(
    rows=2,
    columns=3,
    theme=lc.Themes.Dark
)

# Hotelling's T²
chart1 = dashboard.BarChart(    
    column_index=0,
    row_index=0
)
chart1.set_title("Hotelling's T² Anomaly Scores")
plot_histogram(data, 'hotelling_score', "Hotelling's T²", chart1)

# One-Class SVM
chart2 = dashboard.BarChart(
    column_index=1,
    row_index=0
)
chart2.set_title("One-Class SVM Anomaly Scores")
plot_histogram(data, 'ocsvm_score', 'One-Class SVM', chart2)

# Isolation Forest
chart3 = dashboard.BarChart(
    column_index=2,
    row_index=0
)
chart3.set_title("Isolation Forest Anomaly Scores")
plot_histogram(data, 'iforest_score', 'Isolation Forest', chart3)

# LOF
chart4 = dashboard.BarChart(
    column_index=0,
    row_index=1
)
chart4.set_title("LOF Anomaly Scores")
plot_histogram(data, 'lof_score', 'LOF', chart4)

# ChangeFinder
chart5 = dashboard.BarChart(
    column_index=1,
    row_index=1
)
chart5.set_title("ChangeFinder Anomaly Scores")
plot_histogram(data, 'cf_score', 'ChangeFinder', chart5)

dashboard.open()
