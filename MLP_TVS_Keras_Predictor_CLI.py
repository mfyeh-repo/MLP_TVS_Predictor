#!/usr/bin/env python
# coding: utf-8

# # 🔮 TVS MLP 單筆預測工具 (CLI 模擬版)
# 此筆記本會：
# 1. 載入已訓練的 `mlp_tvs_model.h5`
# 2. 載入標準化器 `scaler_X.pkl` 與 `scaler_y.pkl`
# 3. 模擬命令列式的單筆預測輸入流程，直到輸入 X/x 中斷

# 轉換成 .py 檔
# jupyter nbconvert --to script MLP_TVS_Predictor_CLI.ipynb
# 打包成  .exe 檔指令：
# pyinstaller --onefile MLP_TVS_Predictor_CLI.py

# In[1]:


import numpy as np
from tensorflow.keras.models import load_model
import joblib


# In[2]:


# 載入模型與標準化器
model = load_model("mlp_tvs_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")
print("✅ 模型與標準化器已載入完成")


# In[5]:


# 單筆預測互動迴圈（輸入 x 結束）
print("輸入 Vc_1 和 Ipp_1 進行預測，直到輸入 X 或 x 為止")
while True:
    vcc_input = input("Vc_1 (V): ")
    if vcc_input.strip().lower() == "x":
        print("⛔ 結束預測")
        break

    ipp_input = input("Ipp_1 (A): ")
    if ipp_input.strip().lower() == "x":
        print("⛔ 結束預測")
        break

    try:
        vcc1 = float(vcc_input)
        ipp1 = float(ipp_input)
        X_input = np.array([[vcc1, ipp1]])
        X_scaled = scaler_X.transform(X_input)
        y_scaled_pred = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_scaled_pred)

        print(f"✅ 預測 Vc_2: {y_pred[0][0]:.2f} V")
        print(f"✅ 預測 Ipp_2: {y_pred[0][1]:.2f} A\n")

    except Exception as e:
        print(f"⚠️ 輸入錯誤：{e}\n")


# 轉換成 .py 檔
# jupyter nbconvert --to script MLP_TVS_Predictor_cli_interactive.ipynb

# In[ ]:




