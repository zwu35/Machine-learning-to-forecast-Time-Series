#The code of test of adding Time Series. 
#The code is a sample. In our use, we changed some line for different situation
#We save the results in multiple '.txt' files. We use 'Resultprocee.ipynb' to analyze the result
#The name of the txt file will be changed to save the data from different situation

from gluonts.dataset import common
from gluonts.model import deepar
from gluonts.model import deepstate
from gluonts.model import lstnet
from gluonts.model import seq2seq
from gluonts.model import n_beats
from gluonts.model import canonical
from gluonts.model import deep_factor
from gluonts.evaluation import Evaluator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from itertools import islice

import pandas as pd
import matplotlib.pyplot as plt
import numpy



def validation(a):
    a_10=int(a[2,7],16)
    b_10=int(a[0,1],16)
    if a_10==b_10:
       return("Valid")
    else:
        return("invalid") 


a='1CC0FFEE'
validation(a)