import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

from datetime import datetime as time
from notify import notify

def train():
    
    try:
        directory = './'  # directory where you have downloaded the data CSV files from the competition
        label = 'isFraud'  # name of target variable to predict in this competition
        eval_metric = 'roc_auc'  # Optional: specify that competition evaluation metric is AUC
        save_path = directory + 'ieee-fraud-detection-Models/'  # where to store trained models

        train_data = pd.read_csv(directory+'train_data_fe.csv')

        predictor = TabularPredictor(label=label, eval_metric=eval_metric, path=save_path).fit(
            train_data, presets='best_quality', time_limit=3600
        )

        results = predictor.fit_summary()

        print(results)

        usetime = int(time.now().timestamp() - runStart)
        if usetime > 3600:
            usetime = str(usetime // 3600) + '小时' + str(usetime % 3600)
        msg = "全部模型训练完成, 用时 %s\n%s" % ( usetime, time.now() )
    except Exception as e:
        usetime = int(time.now().timestamp() - runStart)
        if usetime > 3600:
            usetime = str(usetime // 3600) + '小时' + str(usetime % 3600)
        msg = "Error: %s, 用时 %s\n%s" % ( e, usetime, time.now() )
    
    print( msg )
    notify( msg )

if __name__ == '__main__':
    runStart = time.now().timestamp()
    msg = "模型训练开始: %s" % ( time.now() )
    print( msg )
    notify( msg )

    train()