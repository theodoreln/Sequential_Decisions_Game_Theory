# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:39:13 2024

@author: geots
"""

import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import math
import matplotlib.pyplot as plt
import random

def work_vs_study(current_money, current_education_level):
    
    if current_money > 1.8:
        work = 0
        study = 1
    else:
        work = 1
        study = 0
        
    return work, study