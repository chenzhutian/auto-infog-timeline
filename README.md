# Deep Learning-based Auto-Extraction of Extensible Timeline

This project aims at extracting an extensible template from a bitmap timeline infogrpahic using a deep learning model. This project is builted based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Training

### Multi-GPU training

Run the following command in the root of this repo:
```bash
export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_maskrcnn_benchmark/tools/train_net.py --config-file "path/to/config/file.yaml"
```

### Single GPU training

The defualt configuration files that we provide are used for running on 2 GPUs.
In order to run it on a single GPU, there are a few possibilities:

**1. Run the following without modifications**
```bash
python /path_to_maskrcnn_benchmark/tools/train_net.py --config-file "/path/to/config/file.yaml"
```
This should work out of the box and is very similar to what we should do for multi-GPU training.
But the drawback is that it will use much more GPU memory. The reason is that we set in the
configuration files a global batch size that is divided over the number of GPUs. So if we only
have a single GPU, this means that the batch size for that GPU will be 8x larger, which might lead
to out-of-memory errors.

If you have a lot of memory available, this is the easiest solution.

**2. Modify the cfg parameters**

If you experience out-of-memory errors, you can reduce the global batch size. But this means that
you'll also need to change the learning rate (to half), 
and the number of iterations (to double).

### Testing Results
After the training, the programm will conduct an evaluation and output the evaluation results. If you re-run the training without traning the `OUTPUT_DIR`,
the programm will load the last result in `OUTPUT_DIR` directly run the evaluation code by skiping the training.

## Troubleshooting
If you have issues running or compiling this code, we have compiled a list of common issues in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). If your issue is not present there, please feel
free to open a new issue.

## Note

The code in this repo is under active development.
Datasets and demos of Jupyter Notebooks will be provided soon.


## Citing RuleMatrix
```
@ARTICLE{chen19, 
  author    = {Zhutian Chen and
               Yun Wang and
               Qianwen Wang and
               Yong Wang and
               Huamin Qu},
  title     = {Towards Automated Infographic Design: Deep Learning-based Auto-Extraction
               of Extensible Timeline},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  year={2018}, 
  volume={}, 
  number={}, 
  pages={1-1}
}
```