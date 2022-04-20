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
        save_path = directory + 'ieee-fraud-detection-Models_2/'  # where to store trained models

        train_data = pd.read_csv(directory+'ieee_train_full.csv')

        cols = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D10', 'D11', 'D15', 'M1', 'M2', 'M3', 'M4', 'M6', 'M7', 'M8', 'M9', 'V1', 'V3', 'V4', 'V6', 'V8', 'V11', 'V13', 'V14', 'V17', 'V20', 'V23', 'V26', 'V27', 'V30', 'V36', 'V37', 'V40', 'V41', 'V44', 'V47', 'V48', 'V54', 'V56', 'V59', 'V62', 'V65', 'V67', 'V68', 'V70', 'V76', 'V78', 'V80', 'V82', 'V86', 'V88', 'V89', 'V91', 'V107', 'V108', 'V111', 'V115', 'V117', 'V120', 'V121', 'V123', 'V124', 'V127', 'V129', 'V130', 'V136', 'V138', 'V139', 'V142', 'V147', 'V156', 'V160', 'V162', 'V165', 'V166', 'V169', 'V171', 'V173', 'V175', 'V176', 'V178', 'V180', 'V182', 'V185', 'V187', 'V188', 'V198', 'V203', 'V205', 'V207', 'V209', 'V210', 'V215', 'V218', 'V220', 'V221', 'V223', 'V224', 'V226', 'V228', 'V229', 'V234', 'V235', 'V238', 'V240', 'V250', 'V252', 'V253', 'V257', 'V258', 'V260', 'V261', 'V264', 'V266', 'V267', 'V271', 'V274', 'V277', 'V281', 'V283', 'V284', 'V285', 'V286', 'V289', 'V291', 'V294', 'V296', 'V297', 'V301', 'V303', 'V305', 'V307', 'V309', 'V310', 'V314', 'V320', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_31', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'cents', 'addr1_FE', 'card1_FE', 'card2_FE', 'card3_FE', 'P_emaildomain_FE', 'card1_addr1', 'card1_addr1_P_emaildomain', 'card1_addr1_FE', 'card1_addr1_P_emaildomain_FE', 'TransactionAmt_card1_mean', 'TransactionAmt_card1_std', 'TransactionAmt_card1_addr1_mean', 'TransactionAmt_card1_addr1_std', 'TransactionAmt_card1_addr1_P_emaildomain_mean', 'TransactionAmt_card1_addr1_P_emaildomain_std', 'D9_card1_mean', 'D9_card1_std', 'D9_card1_addr1_mean', 'D9_card1_addr1_std', 'D9_card1_addr1_P_emaildomain_mean', 'D9_card1_addr1_P_emaildomain_std', 'D11_card1_mean', 'D11_card1_std', 'D11_card1_addr1_mean', 'D11_card1_addr1_std', 'D11_card1_addr1_P_emaildomain_mean', 'D11_card1_addr1_P_emaildomain_std', 'uid_FE', 'TransactionAmt_uid_mean', 'TransactionAmt_uid_std', 'D4_uid_mean', 'D4_uid_std', 'D9_uid_mean', 'D9_uid_std', 'D10_uid_mean', 'D10_uid_std', 'D15_uid_mean', 'D15_uid_std', 'C1_uid_mean', 'C2_uid_mean', 'C4_uid_mean', 'C5_uid_mean', 'C6_uid_mean', 'C7_uid_mean', 'C8_uid_mean', 'C9_uid_mean', 'C10_uid_mean', 'C11_uid_mean', 'C12_uid_mean', 'C13_uid_mean', 'C14_uid_mean', 'M1_uid_mean', 'M2_uid_mean', 'M3_uid_mean', 'M4_uid_mean', 'M5_uid_mean', 'M6_uid_mean', 'M7_uid_mean', 'M8_uid_mean', 'M9_uid_mean', 'uid_P_emaildomain_ct', 'uid_dist1_ct', 'uid_DT_M_ct', 'uid_id_02_ct', 'uid_cents_ct', 'C14_uid_std', 'uid_C13_ct', 'uid_V314_ct', 'uid_V127_ct', 'uid_V136_ct', 'uid_V309_ct', 'uid_V307_ct', 'uid_V320_ct', 'outsider15', 'isFraud']

        predictor = TabularPredictor(label=label, eval_metric=eval_metric, path=save_path).fit(
            train_data[cols], presets='best_quality', time_limit=3600
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