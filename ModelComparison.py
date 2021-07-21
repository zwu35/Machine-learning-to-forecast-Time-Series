#The code of model comparison 
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


#The 3 TS we use to compare models
df = pd.read_csv('NDDUJN.csv', header=0, index_col=0)

df2 = pd.read_csv('RU10GRTR.csv', header=0, index_col=0)

df3 = pd.read_csv('S5ENRS.csv', header=0, index_col=0)

#Training set
data = common.ListDataset([{
    #The TS we want to use
    "start": df.index[0],
    "target": df.Price[:-100]}

],

    freq="1B")

#Test set. Must be same with training set except not adding '[:-100]' as index
test_data = common.ListDataset([{
    "start": df.index[0],
    "target": df.Price
}],
    freq="1B")


#The model we use
trainer = Trainer(epochs=20)
estimator = deepar.DeepAREstimator(
    freq="1B",prediction_length=20,num_layers=2,context_length=20,num_cells=60,trainer=trainer)

#DeepState
#trainer = Trainer(epochs=20)
#estimator_deep_STATE = deepstate.DeepStateEstimator(
#   freq="1B",prediction_length=20,cardinality=[3],past_length=20,num_layers=1,num_cells=60,use_feat_static_cat=False,
#   trainer=trainer,use_feat_dynamic_real=True)

#DeepFactor
#trainer = Trainer(epochs=30)
#estimator_deep_factor = deep_factor.DeepFactorEstimator(
#   freq="1B",prediction_length=20,trainer=trainer)

#Wavenet
#trainer = Trainer(epochs=50)
#estimator = wavenet.WaveNetEstimator(
#   freq="1B",prediction_length=20,trainer=trainer)

#Gaussian process
#trainer = Trainer(epochs=30)
#estimator_gbforecast = gp_forecaster.GaussianProcessEstimator(
#   freq="1B",prediction_length=20,trainer=trainer,cardinality=1)

#NBEATS
#trainer = Trainer(epochs=20)
#estimator_nbeats = n_beats.NBEATSEstimator(
#   freq="1B",prediction_length=20,trainer=trainer,context_length=20)



#The sampling of MASE. 50 times
for i in range(0,50):
    predictor = estimator.train(training_data=data)
    forecast_it, ts_it = make_evaluation_predictions(test_data, predictor=predictor, num_samples=20000)
    forecasts = list(forecast_it)
    tss = list(ts_it)

    evaluator = Evaluator(quantiles=[0.5], seasonality=2016)
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))
    agg_metrics
    if i==0:
        x=agg_metrics
        MASE=[agg_metrics['MASE']]
    else:
        MASE.append(agg_metrics['MASE'])
    for key in agg_metrics:
        if key=='MASE' :
            x[key]=(x[key]*i+agg_metrics[key])/(i+1)
        elif key=='MAPE':
            x[key]=(x[key]*i+agg_metrics[key])/(i+1)
        elif key=='sMAPE':
            x[key]=(x[key]*i+agg_metrics[key])/(i+1)
        elif key=='RMSE' :
            x[key]=numpy.sqrt((x[key]*x[key]*i+agg_metrics[key]*agg_metrics[key])/(i+1))
        elif key=='NRMSE':
            x[key]=numpy.sqrt((x[key]*x[key]*i+agg_metrics[key]*agg_metrics[key])/(i+1))
    
    print(i)

x=numpy.array(MASE)
#Save the result. The name can be changed
numpy.savetxt('wavnets5MASE.txt',x)
#numpy.savetxt('deepars5MASE.txt',x) example of another name
