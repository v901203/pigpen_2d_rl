# pig_pen_2d_rl

簡短說明：這個專案提供一個 2D 豬舍環境（LiDAR + 障礙物），並以 Stable-Baselines3 的 PPO 來訓練自動駕駛巡檢 agent。



## 快速開始

建立與啟用虛擬環境（建議 Python 3.10）：

```bash
python3 -m venv rl_env
source rl_env/bin/activate
pip install -r requirements.txt
```

如果你已經有專案內的 `rl_env`（本機虛擬環境），可直接：

```bash
source rl_env/bin/activate
pip install -r requirements.txt
```

## 執行

- 訓練（平行版）：

```bash
python train_pigpen.py
```

- 測試（會載入 models/PPO 下的模型並以 `render_mode="human"` 顯示）：

1. 把訓練好的模型 zip 放到 `models/PPO/`，或在 `test_model.py` 調整 `model_path`。
2. 執行：

```bash
python test_model.py
```

## 模型與大檔管理

- 本專案將大型檔案（訓練結果、虛擬環境）排除在 Git 之外。建議做法：
  - 把 `models/PPO/` 打包 (zip) 並上傳到 GitHub Releases、Google Drive 或其他雲端，再在 README 放連結；或
  - 啟用 Git LFS 並將 `models/` 移到 LFS（注意 Github LFS 使用限制與配額）。

```bash
zip -r pigpen_models.zip models/PPO
# 然後手動上傳至 Release / Google Drive
```

## 注意事項

- `pig_pen_env.py` 使用 `pygame` 繪圖；在 headless（無 GUI）環境下請務必在訓練時把 `render_mode=None`。
- `test_model.py` 會嘗試載入 `models/PPO/ppo_pigpen_100k.zip`（可修改路徑）。
- 若要在 GPU 上加速訓練，請安裝對應 CUDA 版本的 `torch`。


---
檔案： [requirements.txt](requirements.txt)、[pig_pen_env.py](pig_pen_env.py)、[train_pigpen.py](train_pigpen.py)、[test_model.py](test_model.py)

## 註記 — 新專題

本倉庫暫停開發，並作為另一開發專案的參考代碼。未來會以 `pig_pen_2d_rlpd` 為新專題名稱，該專題會以本專案為基底進行改良與實驗：
