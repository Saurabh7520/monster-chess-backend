import pandas as pd
import matplotlib.pyplot as plt
import os

def show_dashboard():
    log_file = "training_log.csv"
    
    if not os.path.exists(log_file):
        print(f"Could not find {log_file}. Run the training script first!")
        return
        
    # Read the data
    data = pd.read_csv(log_file)
    
    if len(data) < 5:
        print("Play a few more games before graphing! Need more data.")
        return

    # Calculate a "rolling average" to smooth out the bumpy graph lines
    data['Smoothed_Loss'] = data['Loss'].rolling(window=20, min_periods=1).mean()
    data['Smoothed_Moves'] = data['Moves'].rolling(window=20, min_periods=1).mean()

    # Create the graph window
    plt.figure(figsize=(12, 6))
    
    # 1. Plot the AI's Confusion (Loss)
    plt.subplot(1, 2, 1) # Left graph
    plt.plot(data['Game'], data['Loss'], alpha=0.2, color='red', label='Raw Loss')
    plt.plot(data['Game'], data['Smoothed_Loss'], color='darkred', linewidth=2, label='Trend')
    plt.title("AI Confusion Level (Lower is Better)")
    plt.xlabel("Games Played")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Plot Game Length
    plt.subplot(1, 2, 2) # Right graph
    plt.plot(data['Game'], data['Moves'], alpha=0.2, color='blue', label='Raw Moves')
    plt.plot(data['Game'], data['Smoothed_Moves'], color='darkblue', linewidth=2, label='Trend')
    plt.title("Average Game Length (More moves = smarter defense)")
    plt.xlabel("Games Played")
    plt.ylabel("Number of Moves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_dashboard()