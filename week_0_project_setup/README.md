
**Note: The purpose of the project to explore the libraries and learn how to use them. Not to build a SOTA model.**

## Requirements:

This project uses Python 3.8

Create a virtual env with the following command:

```
conda create --name project-setup python=3.8
conda activate project-setup
```

__NOTE: You can use any environment name you wish. For example a conda environment MLProj38__

Install the requirements:

```
pip install -r requirements.txt
```

## Running

### Training

After installing the requirements, in order to train the model simply run:

```
python train.py
```

### Inference

After training, update the model checkpoint path in the code and run

```
python inference.py
```

### Running notebooks

I am using [Jupyter lab](https://jupyter.org/install) to run the notebooks. 

Since I am using a virtualenv, when I run the command `jupyter lab` it might or might not use the virtualenv.

To make sure to use the virutalenv, run the following commands before running `jupyter lab`

```
conda install ipykernel
python -m ipykernel install --user --name project-setup
pip install ipywidgets
```

Note: After this to use the notebook you still have to install notebooks or lab, through for example `pip install notebook` 


