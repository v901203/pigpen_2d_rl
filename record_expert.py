import numpy as np
import pygame
from pig_pen_env import PigPenEnv

def main():
    # 這裡門寬不要放太大，用真實比例 (1.0) 錄製最好的示範
    env = PigPenEnv(render_mode="human", door_width_scale=1.0)
    obs, _ = env.reset()
    
    print("🎥 開始錄製專家示範！")
    print("請用方向鍵控制小車跑完 8 間豬舍。按 ESC 可以提早結束錄製並存檔。")
    
    expert_obs = []
    expert_actions = []
    
    running = True
    while running:
        action = [0.0, 0.0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1.0
        if keys[pygame.K_DOWN]: action[0] = -1.0
        if keys[pygame.K_LEFT]: action[1] = -1.0
        if keys[pygame.K_RIGHT]: action[1] = 1.0
        
        # 🌟 核心過濾邏輯：只有當你有按按鍵時，才把資料錄下來
        # 避免錄到一堆「停在原地發呆」的資料，導致 AI 學會偷懶
        if action[0] != 0.0 or action[1] != 0.0:
            expert_obs.append(obs)
            expert_actions.append(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"✅ 回合結束！已收集 {len(expert_obs)} 筆專家資料。")
            # 如果你想錄多幾個回合，這裡可以不跳出迴圈
            # 但目前示範錄一次完美通關即可
            break
            
    env.close()
    
    # 將名單存成 Numpy 壓縮檔
    if len(expert_obs) > 0:
        np.savez("expert_data.npz", obs=np.array(expert_obs), actions=np.array(expert_actions))
        print(f"💾 錄製完成！已將 {len(expert_obs)} 步的動作存至 expert_data.npz")
    else:
        print("⚠️ 沒有錄製到任何動作。")

if __name__ == "__main__":
    main()