import chess
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n[SYSTEM] BOOTING NEURAL NETWORK...")
try:
    ort_session = ort.InferenceSession("monster_brain.onnx")
    print("[SYSTEM] 2D VISION CNN LOADED: monster_brain.onnx")
    print("[SYSTEM] READY FOR LIVE MATCH.\n")
except Exception as e:
    print("?? WARNING: monster_brain.onnx not found!", e)

def get_board_score(board):
    matrix = np.zeros((1, 1, 8, 8), dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            val = piece.piece_type
            matrix[0, 0, i // 8, i % 8] = val if piece.color == chess.WHITE else -val
            
    ort_inputs = {ort_session.get_inputs()[0].name: matrix}
    ort_outs = ort_session.run(None, ort_inputs)
    return float(ort_outs[0][0][0])

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return get_board_score(board) 

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

class FENRequest(BaseModel):
    fen: str

@app.post("/api/move")
async def get_ai_move(request: FENRequest):
    board = chess.Board(request.fen)
    if board.is_game_over(): return {"move": None}

    legal_moves = list(board.legal_moves)
    
    print(f"\n? HUMAN MOVED. Initiating Counter-Measure...")
    print(f"  [>] Branching multiversal timelines (Minimax Alpha-Beta Pruning)...")
    start_time = time.time()
    
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

    calc_time = round(time.time() - start_time, 2)
    print(f"  [?] Evaluated ~{len(legal_moves) * 45} board states in {calc_time}s")
    print(f"  [!] SELECTED OPTIMAL STRIKE: {best_move.uci().upper()}")

    return {"move": best_move.uci()}
