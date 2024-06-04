# Time Series Analysis
import lightningchart as lc
import pandas as pd
from datetime import datetime
from scipy.stats import chi2

lc.set_license('my-license-key')

file_path = 'machine_temperature_system_failure.csv'
data = pd.read_csv(file_path)

data['timestamp'] = pd.to_datetime(data['timestamp'])

anomaly_intervals = [
    ["2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000"],
    ["2013-12-15 17:50:00.000000", "2013-12-17 17:00:00.000000"],
    ["2014-01-27 14:20:00.000000", "2014-01-29 13:30:00.000000"],
    ["2014-02-07 14:55:00.000000", "2014-02-09 14:05:00.000000"]
]
def is_anomaly(timestamp):
    for start, end in anomaly_intervals:
        if start <= timestamp <= end:
            return True
    return False

data['anomaly'] = data['timestamp'].apply(lambda x: is_anomaly(x.strftime('%Y-%m-%d %H:%M:%S'))).astype(int)
anomalies = data[data['anomaly'] == 1].copy()
normal_data = data[data['anomaly'] == 0]

data['timestamp_unix'] = data['timestamp'].apply(lambda x: x.timestamp() * 1000)  # Convert to milliseconds
anomalies['timestamp_unix'] = anomalies['timestamp'].apply(lambda x: x.timestamp() * 1000)  # Convert to milliseconds

data['date'] = data['timestamp'].dt.date
daily_mean = data.groupby('date')['value'].mean().reset_index()

daily_mean['date_unix'] = pd.to_datetime(daily_mean['date']).apply(lambda x: x.timestamp() * 1000)

dashboard = lc.Dashboard(
    rows=1,
    columns=2,
    theme=lc.Themes.Dark
)
chart1 = dashboard.ChartXY(column_index=0, row_index=0)
chart1.set_title('Temperature & Given Anomaly Points')
temp_series = chart1.add_line_series()
temp_series.add(data['timestamp_unix'].tolist(), data['value'].tolist())
temp_series.set_name('Temperature')

anomaly_series = chart1.add_point_series()
anomaly_series.add(anomalies['timestamp_unix'].tolist(), anomalies['value'].tolist())
anomaly_series.set_name('Anomaly Points')
anomaly_series.set_point_size(3)
anomaly_series.set_point_color(lc.Color(255, 0, 0))  # Red color

x_axis1 = chart1.get_default_x_axis()
x_axis1.set_tick_strategy('DateTime')
x_axis1.set_scroll_strategy('progressive')
x_axis1.set_interval(start=data['timestamp_unix'].iloc[0], end=data['timestamp_unix'].iloc[-1])
x_axis1.set_title('Date')

y_axis1 = chart1.get_default_y_axis()
y_axis1.set_title('Temperature')
chart1.add_legend()
chart2 = dashboard.ChartXY(column_index=1, row_index=0)
chart2.set_title('Temperature Mean by Day')

mean_temp_series = chart2.add_line_series()
mean_temp_series.add(daily_mean['date_unix'].tolist(), daily_mean['value'].tolist())
mean_temp_series.set_name('Mean Temperature')

x_axis2 = chart2.get_default_x_axis()
x_axis2.set_tick_strategy('DateTime')
x_axis2.set_scroll_strategy('progressive')
x_axis2.set_interval(start=daily_mean['date_unix'].iloc[0], end=daily_mean['date_unix'].iloc[-1])
x_axis2.set_title('Time')

y_axis2 = chart2.get_default_y_axis()
y_axis2.set_title('Temperature')
chart2.add_legend()

dashboard.open()
