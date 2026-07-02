import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import time

# ==========================================
# 1. THE DEEP LEARNING ARCHITECTURE (CNN)
# ==========================================
class CNNMonsterNet(nn.Module):
    def __init__(self):
        super(CNNMonsterNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        return torch.tanh(self.fc2(x))

def board_to_tensor(board):
    matrix = np.zeros(64, dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            val = piece.piece_type
            matrix[i] = val if piece.color == chess.WHITE else -val
    return torch.tensor(matrix).reshape(1, 1, 8, 8)

# ==========================================
# 2. THE MEGA-TRAINING ENGINE
# ==========================================
def mega_train():
    print("🌌 INITIATING MEGA-TRAINING SEQUENCE...")
    
    model = CNNMonsterNet()
    # Using a slightly slower learning rate so it fine-tunes rather than overwrites
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 
    loss_fn = nn.MSELoss()

    model_path = "monster_brain.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("💾 Loaded existing 2D Vision brain. We build from here.")

    # THE OVERNIGHT GRIND
    TARGET_GAMES = 50000 
    gamma = 0.95    
    
    # EPSILON DECAY: Starts at 30% exploration, drops to 1% by the end
    epsilon_start = 0.30
    epsilon_end = 0.01
    epsilon_decay = (epsilon_start - epsilon_end) / TARGET_GAMES

    print(f"🔥 GOAL: {TARGET_GAMES} Games. Epsilon Decay Active.")
    
    start_time = time.time()

    for game in range(1, TARGET_GAMES + 1):
        board = chess.Board()
        memory = [] 
        
        # Calculate current randomness based on how far along we are
        epsilon = max(epsilon_end, epsilon_start - (game * epsilon_decay))

        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            
            if random.random() < epsilon:
                best_move = random.choice(legal_moves)
            else:
                best_move = None
                best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
                
                for move in legal_moves:
                    board.push(move)
                    score = model(board_to_tensor(board)).item()
                    board.pop()

                    if board.turn == chess.WHITE: 
                        if score > best_score:
                            best_score = score
                            best_move = move
                    else: 
                        if score < best_score:
                            best_score = score
                            best_move = move

            board.push(best_move)
            memory.append(board_to_tensor(board))

        # --- GAME OVER REWARDS ---
        result = board.result()
        if result == '1-0': reward = 1.0     
        elif result == '0-1': reward = -1.0  
        else: reward = 0.0                   
        
        # --- BACKPROPAGATION ---
        optimizer.zero_grad()
        loss = 0
        
        for state in reversed(memory):
            prediction = model(state)
            target = torch.tensor([[reward]], dtype=torch.float32)
            loss += loss_fn(prediction, target)
            reward *= gamma 

        loss.backward() 
        optimizer.step() 

        # --- PROGRESS REPORTING ---
        if game % 100 == 0:
            torch.save(model.state_dict(), model_path)
            
            elapsed = time.time() - start_time
            games_per_sec = game / elapsed
            est_remaining_seconds = (TARGET_GAMES - game) / games_per_sec
            est_remaining_hours = est_remaining_seconds / 3600
            
            print(f"🏆 Game {game}/{TARGET_GAMES} | Result: {result} | Epsilon: {epsilon:.3f}")
            print(f"   ⏱️ Speed: {games_per_sec:.1f} games/sec | ETA: {est_remaining_hours:.1f} hours | 💾 Saved!")

    print("✅ MEGA-TRAINING COMPLETE. The Monster is fully grown.")

if __name__ == "__main__":
    mega_train()