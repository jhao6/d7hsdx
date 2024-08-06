This is code for paper "An Accelerated Algorithm for Stochastic Bilevel Optimization under Unbounded Smoothness".

To run data hyper-cleaning on SNLI experiments, you can download the original data from [SNLI Corpus](https://nlp.stanford.edu/projects/snli/), or directly download the **preprocessed version** from [link](https://sendgb.com/zAjLJpQVWca).

Create 'data' directory in the current path by `mkdir data` and put all the data files in `data/` directory.

### Run bilevel [algorithm] on data hyper-cleaning:
```
    python main.py --methods [algorithm] 
```
where the argument 'algorithm'  can  be choosen from [accbo, bo-rep, saba, ma-soba, stocbio, sustain, ttsa, vrbo].
