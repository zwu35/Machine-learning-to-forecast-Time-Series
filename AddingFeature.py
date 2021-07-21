#The code of test of adding feature. 
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



#All the TS we use
df1 = pd.read_csv('NDDUNA.csv', header=0, index_col=0)
df2 = pd.read_csv('NDDUPXJ.csv', header=0, index_col=0)
df3 = pd.read_csv('RU20GRTR.csv', header=0, index_col=0)

#The training set. 'feat_dynamic_real' is the feature we add. 
data = common.ListDataset([{
    #we can change df8 to df3
    "start": df3.index[0],
    "target": df3.Price[:-100],
    #in [], we can add one or more features
    'feat_dynamic_real':[df3.Volume[:-100],df3.SMAVG15[:-100]]
    #'feat_dynamic_real':[df8.Volume[:-100]] Only use Volume
    #'feat_dynamic_real':[df8.SMAVG15[:-100]]  Only use SMAVG(15)
    }
],

    freq="1B")

#The test set. Must be same with training set except not adding '[:-100]' as index
test_data = common.ListDataset([{
    "start": df3.index[0],
    "target": df3.Price,
    'feat_dynamic_real':[df3.Volume,df3.SMAVG15]
}],
    freq="1B")


#The model for training and testing. This can be changed to different models
trainer = Trainer(epochs=20)
estimator = deepar.DeepAREstimator(
    freq="1B",prediction_length=20,num_layers=2,context_length=20,num_cells=60,use_feat_dynamic_real=True,trainer=trainer)

#Using DeepState
#trainer = Trainer(epochs=20)
#estimator_deep_STATE = deepstate.DeepStateEstimator(
#   freq="1B",prediction_length=20,cardinality=[3],past_length=20,num_layers=1,num_cells=60,use_feat_static_cat=False,
#   trainer=trainer,use_feat_dynamic_real=True)


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
        MASE2=[agg_metrics['MASE']]
    else:
        MASE2.append(agg_metrics['MASE'])
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

#Save the result into a .txt file
x=numpy.array(MASE2)

#The name will be changed for different models, TSs and situation
numpy.savetxt('RU20GRTR1VOLUMESMAVG.txt',x)
#numpy.savetxt('RU20GRTR1VOLUME.txt',x) Example of another file name.


