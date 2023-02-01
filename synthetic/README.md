# Synthetic Graph Classification Task

### Create datasets
- Running `main.py` will automatically generate datasets for experiment.


### Hyperparameters
Following Options can be passed to `main.py`.


`--bias:`
Bias level of synthetic graph dataset.
usage example :`--bias 0.1`

`--lr:`
Learning rate for training the model.  
usage example :`--lr 0.001`

`--epochs:`
Number of epochs for training the model.  
usage example :`--epochs 500`

`--intervention:`
Decision on whether model performs intervention or not. 
usage example :`--intervention True`

`--conditional:`
Decision on whether model performs conditional intervention or naive intervention. 
usage example :`--conditional True`

`--lam1:`
Hyperparameters for weight coefficient for KL Loss.  
usage example :`--lam1 1.0`

`--lam2:`
Hyperparameters for weight coefficient for intervention Loss.  
usage example :`--lam2 1.0`