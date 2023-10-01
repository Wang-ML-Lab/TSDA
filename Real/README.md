# TSDA on Real dataset
If you have any questions, feel free to pose an issue or send an email to zihao.xu@rutgers.edu. We are always happy to receive feedback!

The code for TSDA is developed based on [CIDA](https://github.com/hehaodele/CIDA). [CIDA](https://github.com/hehaodele/CIDA) also provides many baseline implementations (e.g., DANN, MDD), which we used for performance comparasion in our paper. Please refer to its [code](https://github.com/hehaodele/CIDA) for details. For baseline GRDA, please refer to this [code](https://github.com/Wang-ML-Lab/GRDA).

## ImageNet-Attribute-DT
### How to Train on ImageNet-Attribute-DT
2. Uncomment the code: "from configs.config_imagenet_11 import opt" in "default_run.py".
3. Comment the code: "from configs.config_cub_18 import opt" in "default_run.py".
3. Run the following code:
    python default_run.py
<!-- ### How to Use the Pretrained Model to Do Inference -->

## CUB-DT
### How to Train on CUB-DT
1. Download the dataset from [here](https://drive.google.com/file/d/15JaioOFq70Sh7zZ5HrVIDTgMB4fC-Uo0/view?usp=drive_link) and unzip under the folder "Real".
2. Uncomment the code: "from configs.config_cub_18 import opt" in "default_run.py".
3. Comment the code: "from configs.config_imagenet_11 import opt" in "default_run.py".
3. Run the following code:
    python default_run.py
<!-- ### How to Use the Pretrained Model to Do Inference -->

## Loss Visualization during Training
We use visdom to visualize. We assume the code is run on a remote gpu machine.

### Change Configurations
Find the config in "config" folder. Choose the config you need and Set "opt.use_visdom" to "True".

### Start a Visdom Server on Your Machine
    python -m visdom.server -p 2000
Now connect your computer with the gpu server and forward the port 2000 to your local computer. You can now go to:
    http://localhost:2000 (Your local address)
to see the visualization during training.
