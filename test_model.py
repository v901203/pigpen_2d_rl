import pygame
from stable_baselines3 import PPO
from pig_pen_env import PigPenEnv

def main():
    # 測試時，開啟 render_mode="human" 來觀看畫面
    # 把門的寬度設回真實比例 1.0 來考驗 AI
    env = PigPenEnv(render_mode="human", door_width_scale=1.0)
    
    # 載入剛才訓練好的模型 
    model_path = "models/PPO/ppo_pigpen_bc_parallel"
    try:
        model = PPO.load(model_path, env=env)
        print("成功載入模型！")
    except Exception as e:
        print(f"找不到模型檔案: {e}")
        return

    # 跑幾個回合看看
    episodes = 5
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        while not done:
            # --- [新增] 處理 Pygame 視窗事件 ---
            # 這樣視窗才不會顯示「沒有回應」，並且允許你隨時點擊 X 關閉測試
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n手動強制結束測試。")
                    env.close()
                    return
            
            # 讓模型根據目前的畫面 (obs) 決定動作 (action)
            # deterministic=True 代表選擇機率最高的最佳動作，不加隨機探索
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            done = terminated or truncated
            
            # (已移除 time.sleep，交由環境內的 clock.tick 控制流暢度)
            
        print(f"回合 {ep+1} 結束 | 總步數: {step_count} | 總獎勵: {total_reward:.2f} | 巡檢進度: {info.get('waypoints_cleared', 0)}/8")

    env.close()

if __name__ == "__main__":
    main()