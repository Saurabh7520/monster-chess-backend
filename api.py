import chess
import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

# ==========================================
# 1. THE BRAIN ARCHITECTURE
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
# 2. LOAD THE 2D VISION BRAIN
# ==========================================
print("🧠 LOADING 2D VISION MONSTER BRAIN...")
model = CNNMonsterNet()
try:
    model.load_state_dict(torch.load("monster_brain.pth", weights_only=True))
    model.eval()
    print("✅ BRAIN LOADED SUCCESSFULLY.")
except FileNotFoundError:
    print("⚠️ WARNING: monster_brain.pth not found!")

# ==========================================
# 3. THE GRANDMASTER OPENING BOOK
# ==========================================
class OpeningBook:
    def __init__(self):
        # Maps the sequence of moves played to an instant, perfect response.
        # Format: "move1,move2,move3" : ["response_A", "response_B"]
        self.repertoire = {
            "": ["e2e4", "d2d4"], # Turn 1
            
            # If White played e4
            "e2e4": ["e7e5", "c7c5"], # Black replies: King's Pawn or Sicilian
            "e2e4,e7e5": ["g1f3", "f1c4"], # White's 2nd move
            "e2e4,c7c5": ["g1f3", "b1c3"], # White vs Sicilian
            "e2e4,e7e5,g1f3": ["b8c6", "g8f6"], # Black defends e5
            "e2e4,e7e5,g1f3,b8c6": ["f1c4", "f1b5"], # White plays Italian or Ruy Lopez
            
            # If White played d4
            "d2d4": ["d7d5", "g8f6"], # Black replies: Queen's Pawn or Indian Defense
            "d2d4,d7d5": ["c2c4", "g1f3"], # White plays Queen's Gambit
            "d2d4,g8f6": ["c2c4", "g1f3"], # White vs Indian Defense
        }

    def get_move(self, board):
        # Convert the history of the board into a comma-separated string
        history = ",".join([move.uci() for move in board.move_stack])
        if history in self.repertoire:
            print("📖 BOOK MOVE DETECTED! Playing instantly.")
            return random.choice(self.repertoire[history])
        return None

# ==========================================
# 4. THE TIME MACHINE (Minimax)
# ==========================================
def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        state = board_to_tensor(board)
        with torch.no_grad():
            return model(state).item()

    legal_moves = list(board.legal_moves)

    if maximizing_player:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha: break
        return min_eval

# ==========================================
# 5. FASTAPI SERVER
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FENRequest(BaseModel):
    fen: str

# Instantiate the library
book = OpeningBook()

@app.post("/api/move")
async def get_ai_move(request: FENRequest):
    board = chess.Board(request.fen)
    if board.is_game_over(): return {"move": None}

    legal_moves = list(board.legal_moves)
    
    # --- STEP 1: CHECK THE OPENING BOOK ---
    book_move = book.get_move(board)
    if book_move and chess.Move.from_uci(book_move) in legal_moves:
        return {"move": book_move} # Return instantly without calculating!

    # --- STEP 2: IF OUT OF BOOK, THINK HARD ---
    print("🧠 OUT OF BOOK. Thinking...")
    best_move = None
    is_white = board.turn == chess.WHITE
    best_score = -float('inf') if is_white else float('inf')
    
    SEARCH_DEPTH = 2

    for move in legal_moves:
        board.push(move)
        score = minimax(board, SEARCH_DEPTH - 1, -float('inf'), float('inf'), not is_white)
        board.pop()

        if is_white:
            if score > best_score:
                best_score = score
                best_move = move
        else:
            if score < best_score:
                best_score = score
                best_move = move

    if best_move is None:
        best_move = random.choice(legal_moves)

    return {"move": best_move.uci()}