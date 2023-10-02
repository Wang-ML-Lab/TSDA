# TSDA on Real dataset
If you have any questions, feel free to pose an issue or send an email to zihao.xu@rutgers.edu. We are always happy to receive feedback!

The code for TSDA is developed based on [CIDA](https://github.com/hehaodele/CIDA). [CIDA](https://github.com/hehaodele/CIDA) also provides many baseline implementations (e.g., DANN, MDD), which we used for performance comparasion in our paper. Please refer to its [code](https://github.com/hehaodele/CIDA) for details. For baseline GRDA, please refer to this [code](https://github.com/Wang-ML-Lab/GRDA).

## Toy
### How to Train on DT-14
```python
    python default_run.py -c config_tree_14 (or)
    python default_run.py --config config_tree_14
```

### How to Train on DT-40
```python
    python default_run.py -c config_tree_40 (or)
    python default_run.py --config config_tree_40
```

## Loss Visualization during Training
We use visdom to visualize. We assume the code is run on a remote gpu machine.

### Change Configurations
Find the config in "config" folder. Choose the config you need and Set "opt.use_visdom" to "True".

### Start a Visdom Server on Your Machine
    python -m visdom.server -p 2000
Now connect your computer with the gpu server and forward the port 2000 to your local computer. You can now go to:
    http://localhost:2000 (Your local address)
to see the visualization during training.
