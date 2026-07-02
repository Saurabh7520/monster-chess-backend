import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# MATCHING GLOBAL ABSOLUTE PATH
LOG_FILE_PATH = r"D:\AI\chess\training_log.csv"

print("========================================")
print("   LIVE DASHBOARD WITH ELO MONITORING   ")
print("========================================")
print(f"Reading logs from: {LOG_FILE_PATH}")

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

while True:
    if not os.path.exists(LOG_FILE_PATH):
        print("[DASHBOARD] Waiting for training_log.csv to be generated...")
        time.sleep(5)
        continue
        
    try:
        data = pd.read_csv(LOG_FILE_PATH)
        row_count = len(data)
        
        if row_count < 2:
            print(f"[DASHBOARD] Only {row_count} games logged. Waiting for more data...")
            time.sleep(5)
            continue
            
        print(f"[DASHBOARD] Updating charts! Total games processed: {row_count}")
        
        # Calculate Simulated Elo
        elo_list = []
        current_elo = 800.0
        for idx, row in data.iterrows():
            if row['Reward'] <= -0.2:
                current_elo += -0.2
            else:
                efficiency_bonus = max(0.1, 1.0 - (row['Moves'] * 0.005))
                current_elo += (5.0 * efficiency_bonus)
            if row['Loss'] < 0.1:
                current_elo += (0.1 - row['Loss']) * 15
            elo_list.append(current_elo)
            
        data['Estimated_Elo'] = elo_list
        data['Smoothed_Loss'] = data['Loss'].rolling(window=20, min_periods=1).mean()
        data['Smoothed_Moves'] = data['Moves'].rolling(window=20, min_periods=1).mean()
        data['Smoothed_Elo'] = data['Estimated_Elo'].rolling(window=20, min_periods=1).mean()
        
        # Plot Loss
        ax1.clear()
        ax1.plot(data['Game'], data['Loss'], alpha=0.15, color='red')
        ax1.plot(data['Game'], data['Smoothed_Loss'], color='darkred', linewidth=2, label='Trend')
        ax1.set_title("AI Confusion Level\n(Lower = Smarter)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot Moves
        ax2.clear()
        ax2.plot(data['Game'], data['Moves'], alpha=0.15, color='blue')
        ax2.plot(data['Game'], data['Smoothed_Moves'], color='darkblue', linewidth=2, label='Trend')
        ax2.set_title("Average Game Length\n(Drops with Aggression)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot Elo
        ax3.clear()
        ax3.plot(data['Game'], data['Estimated_Elo'], alpha=0.2, color='green')
        ax3.plot(data['Game'], data['Smoothed_Elo'], color='darkgreen', linewidth=2, label='Rating')
        ax3.set_title(f"Estimated Elo Progress\n[ Current: {int(current_elo)} ]", fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='lower right')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(5)
        
    except Exception as e:
        print(f"Syncing data stream... ({e})")
        time.sleep(2)