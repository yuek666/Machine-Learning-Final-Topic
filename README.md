# Machine-Learning-Final-Topic

##  環境設定
本專案使用Kaggle進行運行
其中Accelerator選用 GPU T4 x2

##  技術架構

- **文字預處理與 Tokenization**：導入 Hugging Face `transformers` 的 Tokenizer 進行文本斷詞、長度截斷與 Padding 處理。
- **自定義 Dataset 與 DataLoader**：封裝 PyTorch Dataset 實現批次（Batch）數據的高效讀取與 GPU 加速。
- **深度學習分類模型**：基於 `AutoModelForSequenceClassification` 進行下游分類任務的微調（Fine-tuning）。
- **5-Fold 交叉驗證**：將數據切分為 5 個 Fold 輪流進行訓練與驗證，有效防止過擬合並評估模型穩定度。
- **學習率排程優化**：使用 `get_linear_schedule_with_warmup` 結合 AdamW 優化器，穩定模型訓練前期的收斂過程。
- **5-Fold 集成預測 (Ensemble Submission)**：綜合 5 個最佳模型 Fold 的預測機率進行權衡集成，最終產出完全符合 `sample_submission_format.csv` 標準格式的預測結果。



## 專案目錄結構

```text
Machine-Learning-Final-Topic/
├── intput_datas/             
│   ├── vpesg4ktrain1000v1.json
│   ├── vpesg4k_combined_2000.json
│   └── vpesg4k_val_1000.json
├── trained_models/          
│   ├── best_model_fold_1.pt
│   ├── best_model_fold_2.pt
│   └── ... (fold_3 至 fold_5)
├── output_datas/            
│   ├── prediction_ensemble.csv
│   └── prediction_ensemble.json
├── code_ver2-5-1.ipynb      
└── README.md                
```

##  如何使用專案
將code_ver2-5-1.ipynb匯入Kaggle並將input_datas與trained_models放入dataset裡並修改KAGGLE_INPUT_DIR成相對應的值
如匯入trained_models則不用運行訓練cell


##  Code與DataSet
 - Code:     https://www.kaggle.com/code/yuek666/esg-ver2-5-2
 - DataSet:  https://kaggle.com/datasets/53862cbf1541dc2ad2e2af95a05e1d17921b70b2fb59efdeebbd0b7aa2e2ce5d
