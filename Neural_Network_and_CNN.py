# -*- coding: utf-8 -*-
"""
Neural Network and Convolutionary Neural Network
Python script from LinkedIn Learning
"""

# pip install keras          # to install package of keras
# pip install tensorflow     # to install package of tensorflow
# python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose

from keras.datasets import mnist
from keras.preprocessing.image import load_img, array_to_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline    # to display chart in Jupyter Notebook

########## ########## ########## ##########
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
