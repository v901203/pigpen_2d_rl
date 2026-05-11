import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from pig_pen_env import PigPenEnv 

# 平行環境工廠函數
def make_env(rank, seed=0):
    def _init():
        env = PigPenEnv(render_mode=None, door_width_scale=1.0)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def pretrain_with_expert_data(model, data_path="expert_data.npz", epochs=50, batch_size=64):
    """使用 PyTorch 對 PPO 模型進行行為複製 (Behavioral Cloning)"""
    if not os.path.exists(data_path):
        print(f"❌ 找不到專家資料 {data_path}，跳過預訓練。")
        return

    print("🧠 開始讀取專家示範資料進行預訓練 (Behavioral Cloning)...")
    data = np.load(data_path)
    
    # 轉換為 PyTorch Tensor，並放到與模型相同的裝置 (CPU 或 GPU) 上
    obs_tensor = torch.tensor(data["obs"], dtype=torch.float32).to(model.device)
    action_tensor = torch.tensor(data["actions"], dtype=torch.float32).to(model.device)
    
    dataset = TensorDataset(obs_tensor, action_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 針對 PPO 內部的策略神經網路建立優化器
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    
    model.policy.train() # 切換為訓練模式
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_obs, batch_actions in dataloader:
            # 讓神經網路根據專家看到的畫面，預測它會做什麼動作
            # actions_pred 是網路輸出的連續動作預測
            actions_pred, _, _ = model.policy(batch_obs)
            
            # 計算網路預測與人類實際動作的均方誤差 (MSE Loss)
            loss = F.mse_loss(actions_pred, batch_actions)
            
            # 倒傳遞更新神經網路權重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"  > Epoch {epoch+1}/{epochs} | 平均誤差 Loss: {epoch_loss/len(dataloader):.4f}")
            
    print("✅ 預訓練完成！模型已經學會了基礎路線。")

def main():
    num_cpu = 4  
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    models_dir = "models/PPO"
    logdir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # 初始化全新的 PPO 模型
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=logdir,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64
    )

    # 🌟 魔法步驟：在 RL 自己亂練之前，先塞專家資料給它
    pretrain_with_expert_data(model, data_path="expert_data.npz", epochs=100)

    # 開始 PPO 強化學習微調
    # 因為有預訓練打底，訓練步數可以減少，收斂會非常快
    TIMESTEPS = 3000000
    print(f"🚀 開始 PPO 強化訓練 {TIMESTEPS} 步...")
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_PigPen_LiDAR_BC")

    model_path = f"{models_dir}/ppo_pigpen_bc_parallel"
    model.save(model_path)
    print(f"模型已儲存至 {model_path}.zip")

    env.close()

if __name__ == "__main__":
    main()