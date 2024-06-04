# Selected Diagrams Dashboard
import lightningchart as lc
import pandas as pd
from scipy.stats import chi2, gaussian_kde
from datetime import datetime
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import changefinder
from scipy import stats
import numpy as np

lc.set_license('my-license-key')

file_path = 'machine_temperature_system_failure.csv'
data = pd.read_csv(file_path)

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['year'] = data['timestamp'].dt.year
data['month'] = data['timestamp'].dt.month

data['timestamp_unix'] = data['timestamp'].apply(lambda x: x.timestamp() * 1000)

def create_chart(dashboard, title, data, anomalies, column_index, row_index, row_span=1, column_span=1):
    chart = dashboard.ChartXY(
        column_index=column_index,
        row_index=row_index,
        row_span=row_span,
        column_span=column_span
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
    columns=3,
    theme=lc.Themes.Dark
)

chart = dashboard.ChartXY(
    column_index=0,
    row_index=0,
    row_span=1,
    column_span=1
)
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['date'] = data['timestamp'].dt.strftime('%Y-%m-%d')
data['time'] = data['timestamp'].dt.strftime('%H:%M:%S')

pivot_table = data.pivot_table(values='value', index='date', columns='time', aggfunc='mean')
pivot_table = pivot_table.fillna(pivot_table.mean().mean())
heatmap_data = pivot_table.values.T.tolist()
chart.set_title('Temperature & Given Anomaly Points')

series = chart.add_heatmap_grid_series(columns=len(heatmap_data), rows=len(heatmap_data[0]))

series.set_step(x=1000 * 60 * 5, y=1000 * 60 * 60 * 24 * 2)

series.hide_wireframe()
series.set_intensity_interpolation(False)
series.invalidate_intensity_values(heatmap_data)
series.set_palette_colors(
    steps=[
        {'value': 0.0, 'color': lc.Color('blue')},
        {'value': 0.5, 'color': lc.Color('yellow')},
        {'value': 1.0, 'color': lc.Color('red')},
    ],
    look_up_property='value',
    percentage_values=True
)

x_axis = chart.get_default_x_axis()
x_axis.set_tick_strategy('DateTime', time_origin=data['timestamp'].min().timestamp() * 1000)
x_axis.set_title('Time')

y_axis = chart.get_default_y_axis()
y_axis.set_tick_strategy('DateTime', time_origin=data['timestamp'].min().timestamp() * 1000)
y_axis.set_title('Date')



count_data = data.groupby(['year', 'month']).size().unstack(fill_value=0)
mean_temp_data = data.groupby(['year', 'month'])['value'].mean().unstack(fill_value=0)
max_temp_data = data.groupby(['year', 'month'])['value'].max().unstack(fill_value=0)
min_temp_data = data.groupby(['year', 'month'])['value'].min().unstack(fill_value=0)

categories = count_data.index.astype(str).tolist()
data_count = [{'subCategory': str(month), 'values': count_data[month].tolist()} for month in count_data.columns]
data_temp = [{'subCategory': str(month), 'values': mean_temp_data[month].tolist()} for month in mean_temp_data.columns]
data_max_temp = [{'subCategory': str(month), 'values': max_temp_data[month].tolist()} for month in max_temp_data.columns]
data_min_temp = [{'subCategory': str(month), 'values': min_temp_data[month].tolist()} for month in min_temp_data.columns]

temp_chart = dashboard.BarChart(column_index=0, row_index=1, row_span=1)
temp_chart.set_title('Year/Month Mean Temperature')
temp_chart.set_data_stacked(categories, data_temp)
temp_chart.add_legend().add(temp_chart)

temperature_values = data['value'].values

years = data['year'].unique()
months = data['month'].unique()
data_by_year = {year: data[data['year'] == year]['value'] for year in years}
data_by_month = {month: data[data['month'] == month]['value'] for month in months}

densities_by_year = {year: gaussian_kde(values) for year, values in data_by_year.items()}
densities_by_month = {month: gaussian_kde(values) for month, values in data_by_month.items()}
x_vals = np.linspace(min(temperature_values), max(temperature_values), 100)
y_vals_year = {year: density(x_vals) for year, density in densities_by_year.items()}
y_vals_month = {month: density(x_vals) for month, density in densities_by_month.items()}
month_chart = dashboard.ChartXY(column_index=0, row_index=2, row_span=1)
month_chart.set_title('Temperature by Month Distribution')

for month in months:
    series = month_chart.add_positive_area_series()
    series.add(x_vals.tolist(), y_vals_month[month].tolist())
    series.set_name(str(month))

month_chart.add_legend()
month_chart.get_default_x_axis().set_title('Temperature')
month_chart.get_default_y_axis().set_title('Density')

# Anomaly detection models and plots
# Hotelling's T² - Detected Points
mean = data['value'].mean()
std = data['value'].std()
data['anomaly_score'] = ((data['value'] - mean) / std) ** 2
anomaly_threshold = chi2.ppf(q=0.95, df=1)
data['anomaly'] = data['anomaly_score'] > anomaly_threshold
anomalies_hotelling = data[data['anomaly']].copy()
create_chart(dashboard, "Hotelling's T² - Detected Points", data, anomalies_hotelling, 1, 0)

# One-Class SVM
ocsvm_model = OneClassSVM(nu=0.2, gamma=0.001, kernel='rbf')
ocsvm_ret = ocsvm_model.fit_predict(data['value'].values.reshape(-1, 1))
ocsvm_df = data.copy()
ocsvm_df['anomaly'] = [1 if i == -1 else 0 for i in ocsvm_ret]
anomalies_ocsvm = ocsvm_df[ocsvm_df['anomaly'] == 1]
create_chart(dashboard, "One-Class SVM - Detected Points", ocsvm_df, anomalies_ocsvm, 1, 1)

# Isolation Forest
iforest_model = IsolationForest(n_estimators=300, contamination=0.1, max_samples=700, random_state=42)
iforest_ret = iforest_model.fit_predict(data['value'].values.reshape(-1, 1))
iforest_df = data.copy()
iforest_df['anomaly'] = [1 if i == -1 else 0 for i in iforest_ret]
anomalies_iforest = iforest_df[iforest_df['anomaly'] == 1]
create_chart(dashboard, "Isolation Forest - Detected Points", iforest_df, anomalies_iforest, 1, 2)

# Function to calculate anomaly scores for each model
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

# Isolation Forest
chart3 = dashboard.BarChart(
    column_index=2,
    row_index=0,
    row_span=1,
    column_span=1
)
chart3.set_title("Isolation Forest Anomaly Scores")
plot_histogram(data, 'iforest_score', 'Isolation Forest', chart3)

# ChangeFinder - Anomaly Score & Threshold
cf_model = changefinder.ChangeFinder(r=0.002, order=1, smooth=250)
ch_df = pd.DataFrame()
ch_df['value'] = data['value']
ch_df['anomaly_score'] = [cf_model.update(i) for i in ch_df['value']]
ch_score_q1 = stats.scoreatpercentile(ch_df['anomaly_score'], 25)
ch_score_q3 = stats.scoreatpercentile(ch_df['anomaly_score'], 75)
ch_df['anomaly_threshold'] = ch_score_q3 + (ch_score_q3 - ch_score_q1) * 3
ch_df['anomaly'] = ch_df.apply(lambda x: 1 if x['anomaly_score'] > x['anomaly_threshold'] else 0, axis=1)
ch_df['timestamp_unix'] = data['timestamp'].apply(lambda x: x.timestamp() * 1000)
ch_df['timestamp_unix'] = data['timestamp_unix']

chart = dashboard.ChartXY(
    column_index=2,
    row_index=1,
    row_span=1,
    column_span=1
)
chart.set_title("ChangeFinder - Anomaly Score & Threshold")
anomaly_score_series = chart.add_line_series()
anomaly_score_series.add(ch_df['timestamp_unix'].tolist(), ch_df['anomaly_score'].tolist())
anomaly_score_series.set_name('Anomaly Score')

threshold_series = chart.add_line_series()
threshold_series.add(ch_df['timestamp_unix'].tolist(), ch_df['anomaly_threshold'].tolist())
threshold_series.set_name('Threshold')
threshold_series.set_dashed(pattern='Dashed')
threshold_series.set_line_color(lc.Color(255, 0, 0))  # Red color

x_axis = chart.get_default_x_axis()
x_axis.set_tick_strategy('DateTime')
x_axis.set_scroll_strategy('progressive')
x_axis.set_interval(start=ch_df['timestamp_unix'].iloc[0], end=ch_df['timestamp_unix'].iloc[-1])
x_axis.set_title('Date')
y_axis = chart.get_default_y_axis()
y_axis.set_title('Anomaly Score')
chart.add_legend()
iforest_model = IsolationForest(n_estimators=300, contamination=0.1, max_samples=700, random_state=42, n_jobs=-1)
iforest_model.fit(data[['value']])

anomaly_scores = iforest_model.decision_function(data[['value']])
data['anomaly_score'] = anomaly_scores
density = gaussian_kde(anomaly_scores)
x_vals = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 1000)
y_vals = density(x_vals)
chart1 = dashboard.ChartXY(
    column_index=2,
    row_index=2,
    row_span=1,
    column_span=1)
chart1.set_title('SHAP value (impact on model output)')
area_series_pos = chart1.add_area_series()
area_series_neg = chart1.add_area_series()
area_series_pos.add(x_vals.tolist(), y_vals.tolist())
area_series_neg.add(x_vals.tolist(), (-y_vals).tolist())
x_axis = chart1.get_default_x_axis()
x_axis.set_title('Anomaly Score')
y_axis = chart1.get_default_y_axis()
y_axis.set_title('Density')
dashboard.open()






