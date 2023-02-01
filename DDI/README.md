# Drug-Drug Interaction Prediction Task

### Download and Create datasets
- Download Drug-Drug Interaction dataset from https://github.com/isjakewong/MIRACLE/tree/main/MIRACLE/datachem.
    - Since these datasets include duplicate instances in train/validation/test split, merge the train/validation/test dataset.
    - Generate random negative counterparts by sampling a complement set of positive drug pairs as negatives.
    - Split the dataset into 6:2:2 ratio, and create separate csv file for each train/validation/test splits.
- Put each datasets into ``data/raw`` and run ``data.py`` file.
- Then, the python file will create ``{}.pt`` file in ``data/processed``.

### Hyperparameters
Following Options can be passed to `main.py`

`--dataset:`
Name of the dataset. Supported names are: ZhangDDI, and ChChMiner.  
usage example :`--dataset ZhangDDI`

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
