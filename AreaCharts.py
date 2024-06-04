# Area charts
import lightningchart as lc
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

lc.set_license('my-license-key')
file_path = 'machine_temperature_system_failure.csv'
data = pd.read_csv(file_path)

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['year'] = data['timestamp'].dt.year
data['month'] = data['timestamp'].dt.month

temperature_values = data['value'].values

years = data['year'].unique()
months = data['month'].unique()
data_by_year = {year: data[data['year'] == year]['value'] for year in years}
data_by_month = {month: data[data['month'] == month]['value'] for month in months}

# Calculate density
densities_by_year = {year: gaussian_kde(values) for year, values in data_by_year.items()}
densities_by_month = {month: gaussian_kde(values) for month, values in data_by_month.items()}
x_vals = np.linspace(min(temperature_values), max(temperature_values), 100)
y_vals_year = {year: density(x_vals) for year, density in densities_by_year.items()}
y_vals_month = {month: density(x_vals) for month, density in densities_by_month.items()}

dashboard = lc.Dashboard(rows=1, columns=3, theme=lc.Themes.Black)

distribution_chart = dashboard.ChartXY(column_index=0, row_index=0)
distribution_chart.set_title('Temperature Distribution')

distribution_density = gaussian_kde(temperature_values)
y_vals_distribution = distribution_density(x_vals)
series_distribution = distribution_chart.add_positive_area_series()
series_distribution.add(x_vals.tolist(), y_vals_distribution.tolist())

distribution_chart.get_default_x_axis().set_title('Temperature')
distribution_chart.get_default_y_axis().set_title('Density')

year_chart = dashboard.ChartXY(column_index=1, row_index=0)
year_chart.set_title('Temperature by Year Distribution')

for year in years:
    series = year_chart.add_positive_area_series()
    series.add(x_vals.tolist(), y_vals_year[year].tolist())
    series.set_name(str(year))

year_chart.add_legend()
year_chart.get_default_x_axis().set_title('Temperature')
year_chart.get_default_y_axis().set_title('Density')

month_chart = dashboard.ChartXY(column_index=2, row_index=0)
month_chart.set_title('Temperature by Month Distribution')

for month in months:
    series = month_chart.add_positive_area_series()
    series.add(x_vals.tolist(), y_vals_month[month].tolist())
    series.set_name(str(month))

month_chart.add_legend()
month_chart.get_default_x_axis().set_title('Temperature')
month_chart.get_default_y_axis().set_title('Density')

dashboard.open()
