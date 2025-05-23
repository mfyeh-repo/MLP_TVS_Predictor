# MLP_TVS_Predictor
MLP_TVS_Keras_Predictor_Train.ipynb
MLP預測模型的訓練過程
1. 仔入數據(資料)
2. 數據前處理
3. 建立模型(Sequential模型,compile)
4. 訓練模型(fit)
5. 儲存訓練結果mlp_tvs_model.h5 (模型架構+權重)

MLP_TVS_Keras_Predictor_CLI.py
使用以訓練MLP模型來進行預測
1. 載入已訓練的 mlp_tvs_model.h5
2. 載入標準化器 scaler_X.pkl 與 scaler_y.pkl
3. 模擬命令列式的單筆預測輸入流程，直到輸入 X/x 中斷 
