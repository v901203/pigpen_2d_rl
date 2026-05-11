# pig_pen_2d_rl

簡短說明：這個專案提供一個 2D 豬舍環境（LiDAR + 障礙物），並以 Stable-Baselines3 的 PPO 來訓練自動駕駛巡檢 agent。

**重要**：為了讓 repo 保持精簡，`rl_env/`（虛擬環境）、`models/`（訓練結果）與 `logs/` 已被加入 `.gitignore`。請參考下方指示將模型與/或虛擬環境放到本地（或外部儲存）。

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

範例：把本機 models 打包供上傳

```bash
zip -r pigpen_models.zip models/PPO
# 然後手動上傳至 Release / Google Drive
```

## 注意事項

- `pig_pen_env.py` 使用 `pygame` 繪圖；在 headless（無 GUI）環境下請務必在訓練時把 `render_mode=None`。
- `test_model.py` 會嘗試載入 `models/PPO/ppo_pigpen_100k.zip`（可修改路徑）。
- 若要在 GPU 上加速訓練，請安裝對應 CUDA 版本的 `torch`。

## 貢獻

歡迎發 PR 或 issue。若要分享訓練結果，請放置至雲端並回報連結。

---
檔案： [requirements.txt](requirements.txt)、[pig_pen_env.py](pig_pen_env.py)、[train_pigpen.py](train_pigpen.py)、[test_model.py](test_model.py)

## 註記 — 新專題

本倉庫同時作為另一開發專案的參考代碼。未來會以 `pig_pen_rlpd` 為新專題名稱，該專題會以本專案為基底進行改良與實驗（例如資料集處理、domain randomization、以及 RL+PID 混合控制測試）。若你希望我將 `pig_pen_rlpd` 建為獨立 repo，我可以幫你：

- 建立新分支 `pig_pen_rlpd` 並加入實驗代碼範本；
- 或建立新的 GitHub repo 並把所需檔案複製過去（保留 models 與說明）；
- 或在本 repo 下新增 `examples/rlpd/` 目錄放實驗代碼。

請告訴我你想採哪種方式，我會幫你設定。
