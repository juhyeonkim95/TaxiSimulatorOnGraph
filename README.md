# TaxiSimulatorOnGraph

This repository is the official implementation of "Optimizing Large-Scale Fleet Management on a Road
Network using Multi-Agent Deep Reinforcement
Learning with Graph Neural Network".
Because we use the company's data, so if the paper is accepted, 
we will disclose the data after consulting with the company.

## Requirements

We use Deep Graph Library(DGL, https://github.com/dmlc/dgl) 
and OSMnx(https://github.com/gboeing/osmnx)
to handle road network.
For backend of DGL, we use PyTorch.

To install all of the requirements:

```setup
pip install -r requirements.txt
```

## Training and Evaluation

Since real data is not uploaded, only simple grid city case is runnable.
See the jupyter notebook code example for both training and evaluation.