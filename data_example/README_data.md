# Data Readme
The data was provided from *Kakao mobility*, South Korea.
We upload example data in `data_example` folder to just check data format.

**Full data will not be uploaded.**

You should use your own data.

There are total 4 files.

- call data : `seoul_call_data_20181023.csv`
- initial driver distributioin : `seoul_idle_driver_initial_distribution_20181023.csv`
- total driver per time : `seoul_total_driver_per_time_20181023.csv`
- road speed : `seoul_call_data_20181023.csv`

Calls and drivers exist on the edges of the road network.
This edge information in call data file (origin_node_index, destination_node_index) 
and driver distribution (road_id) file 
is represented as a (u, v) format where u and v are osmnx graph id.
You can use your own call and drivers data, but you should convert it to osmnx format.

Speed data is a raw data from government's open data set (https://data.seoul.go.kr/). 
It is written in Korean.
You can use your own speed data, but again you should match it to osmnx road ids.