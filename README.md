# wandb-quick-intro

This is a repository that contains some examples of how I use Weights & Biases to track the experiments of my machine learning projects.

The repository is organized in the following way:
- The notebook `setup_wandb.ipynb` that contains a quick introduction to Weights & Biases, how to initialize it and how to log some metrics.
- The notebook `train_model.ipynb` contains an example of how to train a model logging the metrics to Weights & Biases.
- The notebook `plots.ipynb` contains an example of how to use the Weights & Biases API to plot the results of the experiments.
- The folder hyp_tuning contains an example of how to use a "Sweep" to tune hyperparameters with Weights & Biases.

## How to run the notebooks

I used Poetry as dependency manager. If you have poetry installed, you can skip this step. If you don't have poetry installed, you can find the instructions to install it [here](https://python-poetry.org/docs/).
Once you have poetry installed, you can run:

```
poetry config virtualenvs.in-project true

poetry install 
```

to create a virtualenv and install all the dependencies. 

Then you can run the code in the notebooks.

## Slides

[The slides of the talk are available here](https://speakerdeck.com/lcorbucci/wandb)