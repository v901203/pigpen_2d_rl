import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
# 匯入我們剛才寫好的環境
from pig_pen_env import PigPenEnv 

def make_env(rank, seed=0):
    """
    建立環境的輔助函數 (Factory function)。
    每個獨立的 CPU 進程都會呼叫這個函數來建立自己的專屬環境。
    """
    def _init():
        # 建立環境，保持 headless 模式 (render_mode=None) 以達到最高效能
        env = PigPenEnv(render_mode=None, door_width_scale=1.2)
        # 給予每個環境不同的隨機種子，確保分身們不會跑出完全一模一樣的軌跡
        env.reset(seed=seed + rank)
        return env
    
    set_random_seed(seed)
    return _init

def main():
    # 1. 設定平行環境的數量 (可依據你的電腦 CPU 核心數調整)
    # 一般四核心筆電建議設 4，如果是高階桌機或伺服器可以設 8 或 16
    num_cpu = 4  
    
    print(f"正在啟動 {num_cpu} 個平行環境...")
    
    # 使用 SubprocVecEnv 將環境打包成多進程版本
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # 2. 設定模型儲存與 Log 目錄
    models_dir = "models/PPO"
    logdir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # 3. 初始化 PPO 模型
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=logdir,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64
    )

    # 4. 開始訓練
    # 因為平行訓練收集資料非常快，我們可以把訓練步數拉高到 500k 甚至 100 萬步
    TIMESTEPS = 500000 
    print(f"開始平行訓練 {TIMESTEPS} 步...")
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO_PigPen_LiDAR_Parallel")

    # 5. 儲存訓練好的模型
    model_path = f"{models_dir}/ppo_pigpen_parallel"
    model.save(model_path)
    print(f"模型已儲存至 {model_path}.zip")

    env.close()

# ⚠️ 注意：在 Windows 和某些 Linux 系統中，使用多進程一定要包在 __main__ 裡面
if __name__ == "__main__":
    main()