# All Anomaly Detection Models Diagrams concerning Industrial Machine Anomaly Detection
import lightningchart as lc
import pandas as pd
from scipy.stats import chi2
from datetime import datetime
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import changefinder
from scipy import stats

lc.set_license('my-license-key')

file_path = 'machine_temperature_system_failure.csv'
data = pd.read_csv(file_path)

data['timestamp'] = pd.to_datetime(data['timestamp'])

data['timestamp_unix'] = data['timestamp'].apply(lambda x: x.timestamp() * 1000)

def create_chart(dashboard, title, data, anomalies, column_index, row_index):
    chart = dashboard.ChartXY(
        column_index=column_index,
        row_index=row_index
    )
    chart.set_title(title)

    temp_series = chart.add_line_series()
    temp_series.add(data['timestamp_unix'].tolist(), data['value'].tolist())
    temp_series.set_name('Temperature')

    anomaly_series = chart.add_point_series()
    anomaly_series.add(anomalies['timestamp_unix'].tolist(), anomalies['value'].tolist())
    anomaly_series.set_name('Detected Points')
    anomaly_series.set_point_size(5)
    anomaly_series.set_point_color(lc.Color(255, 0, 0))  # Red color

    x_axis = chart.get_default_x_axis()
    x_axis.set_tick_strategy('DateTime')
    x_axis.set_scroll_strategy('progressive')
    x_axis.set_interval(start=data['timestamp_unix'].iloc[0], end=data['timestamp_unix'].iloc[-1])
    x_axis.set_title('Date')

    y_axis = chart.get_default_y_axis()
    y_axis.set_title('Temperature')

    chart.add_legend()

dashboard = lc.Dashboard(
    rows=3,
    columns=2,
    theme=lc.Themes.Dark
)

# Hotelling's T² - Detected Points
mean = data['value'].mean()
std = data['value'].std()
data['anomaly_score'] = ((data['value'] - mean) / std) ** 2
anomaly_threshold = chi2.ppf(q=0.95, df=1)
data['anomaly'] = data['anomaly_score'] > anomaly_threshold
anomalies_hotelling = data[data['anomaly']].copy()
create_chart(dashboard, "Hotelling's T² - Detected Points", data, anomalies_hotelling, 0, 0)

# One-Class SVM
ocsvm_model = OneClassSVM(nu=0.2, gamma=0.001, kernel='rbf')
ocsvm_ret = ocsvm_model.fit_predict(data['value'].values.reshape(-1, 1))
ocsvm_df = data.copy()
ocsvm_df['anomaly'] = [1 if i == -1 else 0 for i in ocsvm_ret]
anomalies_ocsvm = ocsvm_df[ocsvm_df['anomaly'] == 1]
create_chart(dashboard, "One-Class SVM - Detected Points", ocsvm_df, anomalies_ocsvm, 1, 0)

# Isolation Forest
iforest_model = IsolationForest(n_estimators=300, contamination=0.1, max_samples=700, random_state=42)
iforest_ret = iforest_model.fit_predict(data['value'].values.reshape(-1, 1))
iforest_df = data.copy()
iforest_df['anomaly'] = [1 if i == -1 else 0 for i in iforest_ret]
anomalies_iforest = iforest_df[iforest_df['anomaly'] == 1]
create_chart(dashboard, "Isolation Forest - Detected Points", iforest_df, anomalies_iforest, 0, 1)

# LOF - Detected Points
lof_model = LocalOutlierFactor(n_neighbors=500, contamination=0.07)
lof_ret = lof_model.fit_predict(data['value'].values.reshape(-1, 1))
lof_df = data.copy()
lof_df['anomaly'] = [1 if i == -1 else 0 for i in lof_ret]
anomalies_lof = lof_df[lof_df['anomaly'] == 1]
create_chart(dashboard, "LOF - Detected Points", lof_df, anomalies_lof, 1, 1)

# ChangeFinder - Detected Points
cf_model = changefinder.ChangeFinder(r=0.002, order=1, smooth=250)
ch_df = data.copy()
ch_df['anomaly_score'] = [cf_model.update(i) for i in ch_df['value']]
ch_score_q1 = stats.scoreatpercentile(ch_df['anomaly_score'], 25)
ch_score_q3 = stats.scoreatpercentile(ch_df['anomaly_score'], 75)
ch_df['anomaly_threshold'] = ch_score_q3 + (ch_score_q3 - ch_score_q1) * 3
ch_df['anomaly'] = ch_df.apply(lambda x: 1 if x['anomaly_score'] > x['anomaly_threshold'] else 0, axis=1)
anomalies_ch = ch_df[ch_df['anomaly'] == 1]
create_chart(dashboard, 'ChangeFinder - Detected Points', ch_df, anomalies_ch, 0, 2)

# Variance Based Method - Detected Points
sigma_df = data.copy()
sigma_df['anomaly_threshold_3r'] = mean + 1.5 * std
sigma_df['anomaly_threshold_3l'] = mean - 1.5 * std
sigma_df['anomaly'] = sigma_df.apply(lambda x: 1 if (x['value'] > x['anomaly_threshold_3r']) or (x['value'] < x['anomaly_threshold_3l']) else 0, axis=1)
anomalies_sigma = sigma_df[sigma_df['anomaly'] == 1]
create_chart(dashboard, "Variance Based Method - Detected Points", sigma_df, anomalies_sigma, 1, 2)

dashboard.open()
