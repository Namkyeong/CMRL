# Shift-Robust Molecular Relational Learning with Causal Substructure

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://kdd.org/kdd2023/" alt="Conference">
        <img src="https://img.shields.io/badge/KDD'23-brightgreen" /></a>
    <img src="https://img.shields.io/pypi/l/torch-rechub">
</p>

The official source code for Shift-Robust Molecular Relational Learning with Causal Substructure (CMRL).

## How to run the code
We provide `Readme.md` file for detailed instructions for running our code.

## Overview
Recently, molecular relational learning, whose goal is to predict the interaction behavior between molecular pairs, got a surge of interest in molecular sciences due to its wide range of applications.
In this work, we propose CMRL that is robust to the distributional shift in molecular relational learning by detecting the core substructure that is causally related to chemical reactions.
To do so, we first assume a causal relationship based on the domain knowledge of molecular sciences and construct a structural causal model (SCM) that reveals the relationship between variables.
Based on the SCM, we introduce a novel conditional intervention framework whose intervention is conditioned on the paired molecule.
With the conditional intervention framework, our model successfully learns from the causal substructure and alleviates the confounding effect of shortcut substructures that are spuriously correlated to chemical reactions.
Extensive experiments on various tasks with real-world and synthetic datasets demonstrate the superiority of CMRL over state-of-the-art baseline models.


## Requirements
- Python version: 3.7.10
- Pytorch version: 1.8.1
- torch-geometric version: 1.7.0