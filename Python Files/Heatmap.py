# Heatmap Diagram
import lightningchart as lc
import pandas as pd

lc.set_license('my-license-key')

file_path = 'machine_temperature_system_failure.csv'
data = pd.read_csv(file_path)

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['date'] = data['timestamp'].dt.strftime('%Y-%m-%d')
data['time'] = data['timestamp'].dt.strftime('%H:%M:%S')

pivot_table = data.pivot_table(values='value', index='date', columns='time', aggfunc='mean')
pivot_table = pivot_table.fillna(pivot_table.mean().mean())
heatmap_data = pivot_table.values.T.tolist()

chart = lc.ChartXY(
    theme=lc.Themes.Dark,
    title='Temperature & Given Anomaly Points',
)

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

chart.open()
