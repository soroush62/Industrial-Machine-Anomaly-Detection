# Bar Charts
import lightningchart as lc
import pandas as pd

lc.set_license('my-license-key')

file_path = 'machine_temperature_system_failure.csv'
data = pd.read_csv(file_path)

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['year'] = data['timestamp'].dt.year
data['month'] = data['timestamp'].dt.month

count_data = data.groupby(['year', 'month']).size().unstack(fill_value=0)

mean_temp_data = data.groupby(['year', 'month'])['value'].mean().unstack(fill_value=0)

max_temp_data = data.groupby(['year', 'month'])['value'].max().unstack(fill_value=0)

min_temp_data = data.groupby(['year', 'month'])['value'].min().unstack(fill_value=0)

categories = count_data.index.astype(str).tolist()
data_count = [{'subCategory': str(month), 'values': count_data[month].tolist()} for month in count_data.columns]

data_temp = [{'subCategory': str(month), 'values': mean_temp_data[month].tolist()} for month in mean_temp_data.columns]

data_max_temp = [{'subCategory': str(month), 'values': max_temp_data[month].tolist()} for month in max_temp_data.columns]

data_min_temp = [{'subCategory': str(month), 'values': min_temp_data[month].tolist()} for month in min_temp_data.columns]

dashboard = lc.Dashboard(rows=2, columns=2, theme=lc.Themes.Dark)

count_chart = dashboard.BarChart(column_index=0, row_index=0)
count_chart.set_title('Year/Month Count')
count_chart.set_data_stacked(categories, data_count)
count_chart.add_legend().add(count_chart)

temp_chart = dashboard.BarChart(column_index=1, row_index=0)
temp_chart.set_title('Year/Month Mean Temperature')
temp_chart.set_data_stacked(categories, data_temp)
temp_chart.add_legend().add(temp_chart)

max_temp_chart = dashboard.BarChart(column_index=0, row_index=1)
max_temp_chart.set_title('Year/Month Max Temperature')
max_temp_chart.set_data_stacked(categories, data_max_temp)
max_temp_chart.add_legend().add(max_temp_chart)

min_temp_chart = dashboard.BarChart(column_index=1, row_index=1)
min_temp_chart.set_title('Year/Month Min Temperature')
min_temp_chart.set_data_stacked(categories, data_min_temp)
min_temp_chart.add_legend().add(min_temp_chart)

dashboard.open()
