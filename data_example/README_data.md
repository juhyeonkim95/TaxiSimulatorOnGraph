# Data Readme
We upload example data in `data_example` folder to just check data format.

**Full data will not be uploaded.**

You should use your own data.
Public data (osmnx road data / speed data) is fully uploaded.

Following is explanation for each data file.

(X) means full data has not been uploaded, while (O) means full data has been uploaded.

## CSV files

There are total 4 csv files.

- (X) `seoul_call_data_20181023.csv` : call data at each time stamp / each road.
- (X) `seoul_idle_driver_initial_distribution_20181023.csv` : initial driver distribution at each road.
- (X) `seoul_total_driver_per_time_20181023.csv` : total driver numbers at each time stamp.
- (O) `seoul_call_data_20181023.csv` : road speed at each time stamp.

Calls and drivers exist on the edges of the road network.
This edge information in call data file (origin_node_index, destination_node_index) 
and driver distribution (road_id) file 
is represented as a (u, v) format where u and v are osmnx graph id.
You can use your own call and drivers data, but you should convert it to osmnx format.

Speed data is a raw data from government's open data set (https://data.seoul.go.kr/). 
It is written in Korean.
You can use your own speed data, but again you should match it to osmnx road ids.

## Graphml files

There are total 3 road network data.
- (O) `seoul_rectangular_drive_network_original.graphml` : Original road network of Seoul without any simplification (243,621 edges).
- (O) `seoul_rectangular_drive_network_simplified.graphml` : Simplified road network of Seoul  (13,334 edges).
- (O) `seoul_rectangular_drive_network_simplified_with_speed.graphml` : Simplified road network of Seoul with speed information. Check `Tutorial_SpeedInfoGenerator.ipynb` for details.
