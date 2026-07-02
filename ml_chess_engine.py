import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

# --- 1. BOARD TO MATRIX CONVERSION ---
def board_to_matrix(board):
    matrix = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row = chess.square_rank(square)
        col = chess.square_file(square)
        piece_type_index = piece.piece_type - 1 
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = piece_type_index + color_offset
        matrix[plane][row][col] = 1.0
    return torch.from_numpy(matrix)

# --- 2. THE NEURAL NETWORK ARCHITECTURE ---
class ChessEvaluationNet(nn.Module):
    def __init__(self):
        super(ChessEvaluationNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.output(x))

# --- THE SEARCH FUNCTION ---
# --- THE SEARCH FUNCTION (Alpha-Beta + Neural Network) ---
def evaluate_board_nn(board, net):
    # 1. Immediate game over conditions override the AI's intuition
    if board.is_checkmate():
        # If it is Black's turn and they are in checkmate, White wins (+9999)
        return 9999 if board.turn == chess.BLACK else -9999
    if board.is_game_over():
        return 0 # Draw
        
    # 2. Ask the Neural Network to grade the board
    matrix = board_to_matrix(board).unsqueeze(0)
    with torch.no_grad(): 
        score = net(matrix).item()
        
    return score

def minimax(board, depth, alpha, beta, maximizing_player, net):
    if depth == 0 or board.is_game_over():
        return evaluate_board_nn(board, net)
        
    if maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, net)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break # Prune the bad branches!
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, net)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_best_move_nn(board, net, depth=2):
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    maximizing_player = board.turn == chess.WHITE
    
    if maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, net)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, net)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            
    # Fallback to a random move if something goes wrong
    if best_move is None:
        import random
        best_move = random.choice(list(board.legal_moves))
        
    return best_move
# --- THE UCI HANDSHAKE ---
def uci_mode():
    import sys
    board = chess.Board()
    
    # 1. WAKE UP THE BRAIN
    net = ChessEvaluationNet()
    try:
        # Load the Grandmaster memories!
        net.load_state_dict(torch.load("chess_brain_master.pth", weights_only=True))
        net.eval() # Turn off training mode
    except FileNotFoundError:
        print("Error: Could not find chess_brain_master.pth. Make sure you trained the model!")
        return

    # 2. TALK TO ARENA
    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue
            
        parts = line.split()
        
        if parts[0] == "uci":
            print("id name MTANK_ML_Master")
            print("id author Saurabh Raj")
            print("uciok")
            sys.stdout.flush()
            
        elif parts[0] == "isready":
            print("readyok")
            sys.stdout.flush()
            
        elif parts[0] == "position":
            if "startpos" in line:
                board.set_fen(chess.STARTING_FEN)
                if "moves" in line:
                    moves_index = parts.index("moves")
                    for move_str in parts[moves_index + 1:]:
                        board.push(chess.Move.from_uci(move_str))
                        
        elif parts[0] == "go":
            # Let the Neural Network pick the best move!
            best_move = get_best_move_nn(board, net)
            print(f"bestmove {best_move}")
            sys.stdout.flush()
            
        elif parts[0] == "quit":
            break

if __name__ == "__main__":
    uci_mode()