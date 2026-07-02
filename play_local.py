import chess
from video_api import minimax

board = chess.Board()

print("--- MONSTER BRAIN LOCAL MATCH ---")
while not board.is_game_over():
    print("\nCurrent Board:")
    print(board)
    
    # Player Move
    try:
        user_move = input("\nYour move (e.g., e2e4): ")
        board.push_san(user_move)
    except ValueError:
        print("Invalid move! Try again.")
        continue
        
    if board.is_game_over():
        break
        
    print("\nAI is thinking...")
    # Calculate best move using depth 3
    _, ai_move = minimax(board, depth=3, alpha=-float('inf'), beta=float('inf'), maximizing_player=False)
    
    if ai_move:
        board.push(ai_move)
        print(f"AI played: {ai_move}")

print("Game Over! Result:", board.result())