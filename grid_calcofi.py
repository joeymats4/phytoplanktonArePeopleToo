import pandas as pd
import numpy as np

MILES_PER_DEG_LAT = 69.0
GRID_MILES = 3.0

df = pd.read_csv('Finalized_CalCOFI_Phytoplankton.csv')

min_lat = df['Lat_Dec'].min()
min_lon = df['Lon_Dec'].min()
mean_lat_rad = np.radians(df['Lat_Dec'].mean())

lat_step = GRID_MILES / MILES_PER_DEG_LAT
lon_step = GRID_MILES / (MILES_PER_DEG_LAT * np.cos(mean_lat_rad))

df['_row'] = ((df['Lat_Dec'] - min_lat) / lat_step).astype(int)
df['_col'] = ((df['Lon_Dec'] - min_lon) / lon_step).astype(int)

# Build a stable grid ID mapping sorted by (row, col) with bounding box coords
unique_cells = df[['_row', '_col']].drop_duplicates().sort_values(['_row', '_col']).reset_index(drop=True)
unique_cells['Grid_ID'] = ['grid' + str(i + 1) for i in range(len(unique_cells))]
unique_cells['Lat_Min'] = (min_lat + unique_cells['_row'] * lat_step).round(3)
unique_cells['Lat_Max'] = (min_lat + (unique_cells['_row'] + 1) * lat_step).round(3)
unique_cells['Lon_Min'] = (min_lon + unique_cells['_col'] * lon_step).round(3)
unique_cells['Lon_Max'] = (min_lon + (unique_cells['_col'] + 1) * lon_step).round(3)

df = df.merge(unique_cells, on=['_row', '_col'], how='left')

value_cols = ['T_degC', 'Salnty', 'O2ml_L', 'O2Sat', 'ChlorA', 'Phaeop', 'NO3uM']
grouped = df.groupby(['Date', 'Grid_ID'])[value_cols].mean().reset_index()

bbox_cols = unique_cells[['Grid_ID', 'Lat_Min', 'Lat_Max', 'Lon_Min', 'Lon_Max']]
grouped = grouped.merge(bbox_cols, on='Grid_ID', how='left')

grouped[value_cols] = grouped[value_cols].round(3)

grouped = grouped[['Date', 'Grid_ID', 'Lat_Min', 'Lat_Max', 'Lon_Min', 'Lon_Max'] + value_cols]
grouped.to_csv('Gridded_CalCOFI_Phytoplankton.csv', index=False)

print(f'Done. {len(grouped)} rows, {grouped["Grid_ID"].nunique()} unique grid cells.')
print(grouped.head())
