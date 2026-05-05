from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chess import Board, Coordinate, Piece, PieceType, Side
from src.state import GameState, PieceValue, SEARCH_DEPTH


def test_left_depth_decrements() -> None:
    """apply_move should decrement left_depth by 1."""
    board = Board.from_pieces([
        Piece(PieceType.KING, Side.RED, Coordinate(4, 0)),
        Piece(PieceType.KING, Side.BLACK, Coordinate(4, 9)),
    ])
    state = GameState.from_board(board, Side.RED)

    piece = state.board.get_piece_at(Coordinate(4, 0))
    assert piece is not None

    next_state = state.apply_move(piece, Coordinate(4, 1), deep_copy=True)

    assert state.left_depth == SEARCH_DEPTH
    assert next_state.left_depth == SEARCH_DEPTH - 1


def test_apply_move_deep_copy_behavior() -> None:
    """deep_copy=False shares board, deep_copy=True creates a new board."""
    # deep_copy=False: board should be shared
    board_shared = Board.from_pieces([
        Piece(PieceType.KING, Side.RED, Coordinate(4, 0)),
        Piece(PieceType.KING, Side.BLACK, Coordinate(4, 9)),
    ])
    state_shared = GameState.from_board(board_shared, Side.RED)
    shared_piece = state_shared.board.get_piece_at(Coordinate(4, 0))
    assert shared_piece is not None

    child_shared = state_shared.apply_move(shared_piece, Coordinate(4, 1), deep_copy=False)

    assert child_shared.board is state_shared.board
    assert state_shared.board.get_piece_at(Coordinate(4, 0)) is None
    assert state_shared.board.get_piece_at(Coordinate(4, 1)) is not None

    # deep_copy=True: board should be independent
    board_copied = Board.from_pieces([
        Piece(PieceType.KING, Side.RED, Coordinate(4, 0)),
        Piece(PieceType.KING, Side.BLACK, Coordinate(4, 9)),
    ])
    state_copied = GameState.from_board(board_copied, Side.RED)
    copied_piece = state_copied.board.get_piece_at(Coordinate(4, 0))
    assert copied_piece is not None

    child_copied = state_copied.apply_move(copied_piece, Coordinate(4, 1), deep_copy=True)

    assert child_copied.board is not state_copied.board
    assert state_copied.board.get_piece_at(Coordinate(4, 0)) is not None
    assert state_copied.board.get_piece_at(Coordinate(4, 1)) is None
    assert child_copied.board.get_piece_at(Coordinate(4, 0)) is None
    assert child_copied.board.get_piece_at(Coordinate(4, 1)) is not None


def test_apply_move_records_last_move_correctly() -> None:
    """apply_move should record from_pos/to_pos/captured correctly."""
    board = Board.from_pieces([
        Piece(PieceType.KING, Side.RED, Coordinate(4, 0)),
        Piece(PieceType.KING, Side.BLACK, Coordinate(4, 9)),
    ])
    state = GameState.from_board(board, Side.RED)

    piece = state.board.get_piece_at(Coordinate(4, 0))
    assert piece is not None

    next_state = state.apply_move(piece, Coordinate(4, 1), deep_copy=True)
    assert next_state.last_move is not None

    assert next_state.last_move.from_pos == Coordinate(4, 0)
    assert next_state.last_move.to_pos == Coordinate(4, 1)
    assert next_state.last_move.captured is None


def test_capture_updates_score() -> None:
    """When a capture occurs, score should be updated with captured piece value."""
    board = Board.from_pieces([
        Piece(PieceType.KING, Side.RED, Coordinate(4, 0)),
        Piece(PieceType.KING, Side.BLACK, Coordinate(4, 9)),
        Piece(PieceType.CANNON, Side.RED, Coordinate(1, 2)),
        Piece(PieceType.SOLDIER, Side.BLACK, Coordinate(1, 7)),  # cannon screen
        Piece(PieceType.HORSE, Side.BLACK, Coordinate(1, 9)),
    ])
    state = GameState.from_board(board, Side.RED)

    cannon = state.board.get_piece_at(Coordinate(1, 2))
    assert cannon is not None

    next_state = state.apply_move(cannon, Coordinate(1, 9), deep_copy=True)

    assert next_state.score == state.score + PieceValue[PieceType.HORSE]
    assert next_state.last_move is not None
    assert next_state.last_move.captured is not None
    assert next_state.last_move.captured.piece_type == PieceType.HORSE


def test_recover_restores_board() -> None:
    """recover should restore board to state before apply_move."""
    board = Board.from_pieces([
        Piece(PieceType.KING, Side.RED, Coordinate(4, 0)),
        Piece(PieceType.KING, Side.BLACK, Coordinate(4, 9)),
        Piece(PieceType.CANNON, Side.RED, Coordinate(1, 2)),
        Piece(PieceType.SOLDIER, Side.BLACK, Coordinate(1, 7)),  # cannon screen
        Piece(PieceType.HORSE, Side.BLACK, Coordinate(1, 9)),
    ])
    state = GameState.from_board(board, Side.RED)

    cannon = state.board.get_piece_at(Coordinate(1, 2))
    assert cannon is not None

    child = state.apply_move(cannon, Coordinate(1, 9), deep_copy=False)

    # Ensure move has happened on shared board.
    assert state.board.get_piece_at(Coordinate(1, 2)) is None
    moved_piece = state.board.get_piece_at(Coordinate(1, 9))
    assert moved_piece is not None and moved_piece.side == Side.RED

    child.recover()

    restored_cannon = state.board.get_piece_at(Coordinate(1, 2))
    restored_horse = state.board.get_piece_at(Coordinate(1, 9))

    assert restored_cannon is not None
    assert restored_cannon.piece_type == PieceType.CANNON
    assert restored_cannon.side == Side.RED

    assert restored_horse is not None
    assert restored_horse.piece_type == PieceType.HORSE
    assert restored_horse.side == Side.BLACK


def run_all_tests() -> None:
    tests = [
        ("left_depth decrements", test_left_depth_decrements),
        ("deep_copy behavior", test_apply_move_deep_copy_behavior),
        ("last_move record", test_apply_move_records_last_move_correctly),
        ("capture updates score", test_capture_updates_score),
        ("recover restores board", test_recover_restores_board),
    ]

    failures = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"✓ {name}")
        except (AssertionError, Exception) as e:
            print(f"✗ {name}: {e}")
            failures += 1

    if failures == 0:
        print("All state tests passed! ✓")
    else:
        print(f"{failures} state test(s) failed! ✗")


if __name__ == "__main__":
    run_all_tests()
