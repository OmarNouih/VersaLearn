from __future__ import absolute_import, division, print_function
import tkinter as tk
from tkinter import ttk, simpledialog, filedialog, messagebox
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor , DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR , SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error , mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix , r2_score
import seaborn as sns
from pandastable.data import TableModel
from pandastable import images, util, config
from pandastable.dialogs import *
import os
import platform
import logging
import textwrap
import sv_ttk
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.base import check_array
