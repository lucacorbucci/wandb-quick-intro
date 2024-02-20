# Hyperparameter tuning with Weights & Biases

The file `train.py` contains a simple example of how to use Weights & Biases to tune hyperparameters. 

In this case we used the Mnist dataset, the code used to train the model is the one of the [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py).

Weights and Biases offer a tool called "Sweeps" that allows you to run multiple hyperparameter configurations just by defining a .yaml file. 
The .yaml file must contains:
- The hyperparameters that we want to tune with the range of values that we want to try.
- The objective metric that we want to optimize.
- The method of optimization (random, grid, bayesian).
- The name of the script to run

Inside your script you need to use argpase to pass the hyperparameters to the script from the command line.

In this folder you can also find a file `run_sweep.sh` that contains the command to run the sweep.

To run the sweep you need to run the following command:
```bash
sh run_sweep.sh
```
After you run this command, you'll find a new sweep in your Weights & Biases dashboard and you will be able to compare results. 
W&B offers a nice visualization of the results with a pareto frontier that can be used to pick the parameters that best fit your model.

Make sure you created the ../data folder to store the training dataset. 