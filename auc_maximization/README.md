This is code for paper "An Accelerated Algorithm for Stochastic Bilevel Optimization under Unbounded Smoothness".

To run AUC maximation, you should download sentiment140 first, or directly download the **preprocessed version** from [link](https://sendgb.com/zAjLJpQVWca).

Create 'data' directory in the current path by `mkdir data` and put all the data files in `data/` directory.

### Requirements
python 3.9,
numpy,
sklearn,
Pytorch>2.0

### Run bilevel [algorithm] on AUC maximization:
```
    python main.py --methods [algorithm] 
```
where the argument 'algorithm'  can  be chosen from [accbo, bo-rep, saba, ma-soba, stocbio, sustain, ttsa, vrbo]. 
