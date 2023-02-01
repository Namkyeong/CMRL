# Graph Similarity Learning Task

### Download and Create datasets
- Download OpenSSL and ffmpeg dataset from https://github.com/cszhangzhen/H2MN.
- Download AIDS, LINUX, and IMDB dataset from https://github.com/yunshengb/SimGNN.
- Put each datasets into ``./datasets``.

### Hyperparameters
Following Options can be passed to `main_regression.py` and `main_classification.py`

`--dataset:`
Name of the dataset. Supported names are: AIDS700nef, LINUX, IMDBMulti, and ffmpeg_min50 openssl_min50.  
usage example :`--dataset AIDS700nef`

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