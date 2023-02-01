# Molecular Interaction Prediction Task

### Download and Create datasets
- Download Chromophore dataset from https://figshare.com/articles/dataset/DB_for_chromophore/12045567/2, and leave only **Absorption max (nm)**,  **Emission max (nm)**, and **Lifetime (ns)** column.
    - Make separate csv file for each column, and erase the NaN values for each column.
    - We log normalize the target value for **Lifetime (ns)** data due to its high skewness.
- Download Solvation Free Energy datasets from https://www.sciencedirect.com/science/article/pii/S1385894721008925#appSB, and create the dataset based on the **Source_all** column in the excel file.
    - Make separate csv file for each data source.
- Put each datasets into ``data/raw`` and run ``data.py`` file.
- Then, the python file will create ``{}.pt`` file in ``data/processed``.

### Hyperparameters
Following Options can be passed to `main.py`

`--dataset:`
Name of the dataset. Supported names are: chr_abs, chr_emi, chr_emi, mnsol, freesol, compsol, abraham, and combisolv.  
usage example :`--dataset chr_abs`

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