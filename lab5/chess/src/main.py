"""Main module for human vs AI chess game."""

import sys
from copy import deepcopy
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chess import Board, Coordinate, Side
import src.minmax as minmax


def play_game(board: Board, user_side: Side) -> None:
    """
    Play a chess game between user and AI (minmax).

    Args:
        board: Initial board configuration
        user_side: User's color (RED or BLACK)
    """
    ai_side = Side.BLACK if user_side == Side.RED else Side.RED
    current_turn = Side.RED

    # Record: list of Board snapshots (deep copies so later moves don't mutate history)
    history: list[Board] = []
    history.append(deepcopy(board))

    while True:
        print(board)
        print(f"当前轮到: {'用户' if current_turn == user_side else 'AI'} ({current_turn.value})\n")

        if current_turn == user_side:
            # User's turn
            while True:
                try:
                    user_input = input("请输入您的走法: ").strip()

                    if user_input == "Oh no!":
                        if len(history) >= 3:
                            history.pop()
                            history.pop()
                            saved = history[-1]
                            board._pieces.clear()
                            board._pieces.update(saved._pieces)
                            print("哈哈，服了吧\n")
                            print(board)
                            continue
                        else:
                            print("无法悔棋，还没有走够足够的步数\n")
                            continue

                    if user_input == "I'm a cheater":
                        from_pos, to_pos = minmax.solve(board, user_side)
                        print(f"AI建议: {from_pos} -> {to_pos}\n")
                        continue

                    if len(user_input) != 4:
                        print("输入格式错误，请输入4个字符（如 e0e1）\n")
                        continue

                    from_str = user_input[:2]
                    to_str = user_input[2:4]

                    from_pos = Coordinate.from_string(from_str)
                    to_pos = Coordinate.from_string(to_str)

                    piece = board.get_piece_at(from_pos)

                    if piece is None:
                        print(f"错误: {from_pos} 处没有棋子\n")
                        continue

                    if piece.side != user_side:
                        print(f"错误: {from_pos} 处不是您的棋子\n")
                        continue

                    if not piece.is_move_legal(from_pos, to_pos, board):
                        print(f"错误: {from_pos} 的棋子不能走到 {to_pos}\n")
                        continue

                    # Execute user's move
                    captured = board.move_piece(piece, to_pos, deepcopy=False)
                    print(f"您走: {from_pos} -> {to_pos}")
                    if captured:
                        print(f"吃掉了对方的 {captured.side.value}{captured.piece_type.value}\n")
                    else:
                        print()
                    break

                except ValueError as e:
                    print(f"坐标错误: {e}\n")
                    continue

        else:
            # AI's turn
            from_pos, to_pos = minmax.solve(board, ai_side)

            piece = board.get_piece_at(from_pos)
            if piece is not None:
                captured = board.move_piece(piece, to_pos, deepcopy=False)
                print(f"AI走 {piece.side.value}{piece.piece_type.value}: {from_pos} -> {to_pos}")
                if captured:
                    print(f"吃掉了您的 {captured.side.value}{captured.piece_type.value}\n")
                else:
                    print()
            else:
                print("错误: AI计算出错误的走法\n")
                return

        # Record board snapshot after this move (deep copy to isolate from mutating board)
        history.append(deepcopy(board))

        # Check if game is over (king captured)
        red_king = any(
            p.piece_type.name == "KING" and p.side == Side.RED
            for p in board._pieces.values()
        )
        black_king = any(
            p.piece_type.name == "KING" and p.side == Side.BLACK
            for p in board._pieces.values()
        )

        if not red_king:
            print("\n黑方赢了！")
            print(board)
            break

        if not black_king:
            print("\n红方赢了！")
            print(board)
            break

        # Switch turn
        current_turn = Side.BLACK if current_turn == Side.RED else Side.RED

def watch_ai_game(board: Board) -> None:
    """
    AI vs AI exhibition match. Press Enter to advance one move, 'q' to quit.

    Args:
        board: Initial board configuration
    """
    current_turn = Side.RED

    while True:
        print(board)
        print(f"当前: {'红方' if current_turn == Side.RED else '黑方'} ({current_turn.value})")

        user_input = input("按回车继续，输入 q 退出: ").strip()
        if user_input.lower() == "q":
            print("已退出观战。")
            break

        from_pos, to_pos = minmax.solve(board, current_turn)
        piece = board.get_piece_at(from_pos)
        if piece is not None:
            captured = board.move_piece(piece, to_pos, deepcopy=False)
            print(f"{piece.side.value}{piece.piece_type.value}: {from_pos} -> {to_pos}")
            if captured:
                print(f"吃掉了 {captured.side.value}{captured.piece_type.value}\n")
            else:
                print()
        else:
            print("错误: AI计算出错误的走法\n")
            break

        # Check king capture
        red_king = any(
            p.piece_type.name == "KING" and p.side == Side.RED
            for p in board._pieces.values()
        )
        black_king = any(
            p.piece_type.name == "KING" and p.side == Side.BLACK
            for p in board._pieces.values()
        )

        if not red_king:
            print("\n黑方赢了！")
            print(board)
            break
        if not black_king:
            print("\n红方赢了！")
            print(board)
            break

        current_turn = Side.BLACK if current_turn == Side.RED else Side.RED


if __name__ == "__main__":
    print("选择模式:")
    print("  1 - 人机对战 (红方)")
    print("  2 - AI观战")
    choice = input("请输入: ").strip()
    board = Board()
    if choice == "2":
        watch_ai_game(board)
    else:
        play_game(board, user_side=Side.RED)
