# TaxiSimulatorOnGraph
![teasure](assets/seoul.png)

This repository is the official implementation of "Optimizing Large-Scale Fleet Management on a Road
Network using Multi-Agent Deep Reinforcement
Learning with Graph Neural Network". (preprint, work in progress)
Because we use the company's data, we will disclose the data after consulting with the company.

## Requirements

We use [Deep Graph Library](https://github.com/dmlc/dgl) (DGL)
and [OSMnx](https://github.com/gboeing/osmnx)
to handle road network.
For backend of DGL, we use PyTorch.

To install all of the requirements:

```setup
pip install -r requirements.txt
```

## Training and Evaluation

Since real data is not uploaded, only simple grid city case is runnable.
See the jupyter notebook code example for both training and evaluation.
- `Tutorial_10by10GridCity.ipynb` runs simulation in simple 10 by 10 grid city.
- `Tutorial_GraphSimplification.ipynb` performs graph simplification which is required to run simulation 
with large real data (`Tutorial_RealCity.ipynb`).
- `Tutorial_RealCity.ipynb` runs simulation in Seoul with real call data. 
As mentioned before, because data has not been uploaded, you cannot run this code right now.

## Visualization

You can export Q value of the road at each time stamp to SVG file by enabling `export_q_value_image` option in evaluation function.
Following is the video that shows Q values of roads in Seoul at each time stamp in a single day.
Note that red means higher value and green means lower value.

![visualization](assets/teasure.gif)