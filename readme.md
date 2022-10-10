# Simultaneous spectral recovery and CMOS micro-LED holography with an untrained deep neural network

This repository provides code and datasets used for the Optica paper (Kang and de Cea et al, Optica (2022)).

- source: a folder containing an experimental measurement using a commerical LED and measured spectrum of the LED.
- figure: a folder containing visualizations of reconstructions.
- rec: a folder containing reconstruction files.
- train.ipynb: a Jupyter notebook with all the required procedures, including data loading, model definition, and iterative optimization.
- models.py: a Python file with model implementation for untrained neural networks.
- physical.py: a Python file with functions for implementing a physical forward model with multiple wavelengths.
- utils.py: a Python file with auxiliary functions.
- losses.py: a Python file with various kinds of training loss functions and regularizations.

Any questions regarding the code should be directed to Iksung Kang (iksung.kang@alum.mit.edu).
