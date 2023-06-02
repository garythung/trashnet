# trashnet
Code (only for the convolutional neural network) and dataset for mine and [Mindy Yang](http://github.com/yangmindy4)'s final project for [Stanford's CS 229: Machine Learning class](http://cs229.stanford.edu). Our paper can be found [here](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf). The convolutional neural network results on the poster are dated since we continued working after the end of the quarter and  were able to achieve around 75% test accuracy (with 70/13/17 train/val/test split) after changing the weight initialization to the Kaiming method.

## Dataset
This repository contains the dataset that we collected. The dataset spans six classes: glass, paper, cardboard, plastic, metal, and trash. Currently, the dataset consists of 2527 images:
- 501 glass
- 594 paper
- 403 cardboard
- 482 plastic
- 410 metal
- 137 trash

The pictures were taken by placing the object on a white posterboard and using sunlight and/or room lighting. The pictures have been resized down to 512 x 384, which can be changed in `data/constants.py` (resizing them involves going through step 1 in usage). The devices used were Apple iPhone 7 Plus, Apple iPhone 5S, and Apple iPhone SE.

The size of the original dataset, ~3.5GB, exceeds the git-lfs maximum size so it has been uploaded to Google Drive. If you are planning on using the Python code to preprocess the original dataset, then download `dataset-original.zip` from the link below and place the unzipped folder inside of the `data` folder.

**If you are using the dataset, please give a citation of this repository. The dataset can be downloaded [here](https://huggingface.co/datasets/garythung/trashnet).**

## Installation
### Lua setup
We wrote code in [Lua](http://lua.org) using [Torch](http://torch.ch); you can find installation instructions
[here](http://torch.ch/docs/getting-started.html). You'll need the following Lua packages:

- [torch/torch7](http://github.com/torch/torch7)
- [torch/nn](http://github.com/torch/nn)
- [torch/optim](http://github.com/torch/optim)
- [torch/image](http://github.com/torch/image)
- [torch/gnuplot](http://github.com/torch/gnuplot)

After installing Torch, you can install these packages by running the following:

```bash
# Install using Luarocks
luarocks install torch
luarocks install nn
luarocks install optim
luarocks install image
luarocks install gnuplot
```

We also need [@e-lab](http://github.com/e-lab)'s [weight-init module](http://github.com/e-lab/torch-toolbox/blob/master/Weight-init/weight-init.lua), which is already included in this repository.

### CUDA support
Because training takes awhile, you will want to use a GPU to get results in a reasonable amount of time. We used CUDA with a GTX 650 Ti with CUDA. To enable GPU acceleration with CUDA, you'll first need to install CUDA 6.5 or higher. Find CUDA installations [here](http://developer.nvidia.com/cuda-downloads).

Then you need to install following Lua packages for CUDA:
- [torch/cutorch](http://github.com/torch/cutorch)
- [torch/cunn](http://github.com/torch/cunn)

You can install these packages by running the following:

```bash
luarocks install cutorch
luarocks install cunn
```

### Python setup
Python is currently used for some image preprocessing tasks. The Python dependencies are:
- [NumPy](http://numpy.org)
- [SciPy](http://scipy.org)

You can install these packages by running the following:

```bash
# Install using pip
pip install numpy scipy
```

## Usage

### Step 1: Prepare the data
Unzip `data/dataset-resized.zip`.

If adding more data, then the new files must be enumerated properly and put into the appropriate folder in `data/dataset-original` and then preprocessed. Preprocessing the data involves deleting the `data/dataset-resized` folder and then calling `python resize.py` from `trashnet/data`. This will take around half an hour.

### Step 2: Train the model
TODO

### Step 3: Test the model
TODO

### Step 4: View the results
TODO

## Contributing
1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request

## Acknowledgments
- Thanks to the Stanford CS 229 autumn 2016-2017 teaching staff for a great class!
- [@e-lab](http://github.com/e-lab) for their [weight-init Torch module](http://github.com/e-lab/torch-toolbox/blob/master/Weight-init/weight-init.lua)

## TODOs
- finish the Usage portion of the README
- add specific results (and parameters used) that were achieved after the CS 229 project deadline
- add saving of confusion matrix data and creation of graphic to `plot.lua`
- rewrite the data preprocessing to only reprocess new images if the dimensions have not changed
