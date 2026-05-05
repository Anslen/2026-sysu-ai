"""Test file for Chinese Chess class"""

from typing import Optional
from pathlib import Path
import sys

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chess import Board, Coordinate, PieceType, Side, Piece
from src.state import MoveRecord


def test_initial_board() -> None:
    """Test initial board setup"""
    board = Board()
    piece_count = len(board._pieces)
    assert piece_count == 32, "Initial pieces should be 32"


def test_king_movement() -> None:
    """Test king/general movement"""
    board = Board()

    red_king = board.get_piece_at(Coordinate(4, 0))
    assert red_king is not None and red_king.piece_type == PieceType.KING
    moves = list(red_king.get_possible_moves(board))
    assert len(moves) == 1, "Red king should have 1 possible move"

    black_king = board.get_piece_at(Coordinate(4, 9))
    assert black_king is not None and black_king.piece_type == PieceType.KING
    moves = list(black_king.get_possible_moves(board))
    assert len(moves) == 1, "Black general should have 1 possible move"


def test_advisor_movement() -> None:
    """Test advisor movement"""
    board = Board()

    red_advisor = board.get_piece_at(Coordinate(3, 0))
    assert red_advisor is not None and red_advisor.piece_type == PieceType.ADVISOR
    moves = list(red_advisor.get_possible_moves(board))
    assert len(moves) == 1, "Red advisor should have 1 possible move"


def test_chariot_movement() -> None:
    """Test chariot movement"""
    board = Board()

    red_chariot = board.get_piece_at(Coordinate(0, 0))
    assert red_chariot is not None and red_chariot.piece_type == PieceType.CHARIOT
    moves = list(red_chariot.get_possible_moves(board))
    assert len(moves) == 2, "Red chariot should have 2 possible moves"


def test_piece_movement() -> None:
    """Test piece movement execution"""
    board = Board()

    king_piece = board.get_piece_at(Coordinate(4, 0))
    assert king_piece is not None
    captured: Optional[Piece] = board.move_piece(king_piece, Coordinate(4, 1))
    assert captured is None, "Moving to empty square should not capture"

    moved_king = board.get_piece_at(Coordinate(4, 1))
    assert moved_king is not None and moved_king.side == Side.RED
    assert board.get_piece_at(Coordinate(4, 0)) is None, "Original position should be empty"


def test_elephant_movement() -> None:
    """Test elephant movement (check elephant eye)"""
    board = Board()

    red_elephant = board.get_piece_at(Coordinate(2, 0))
    assert red_elephant is not None and red_elephant.piece_type == PieceType.ELEPHANT
    moves = list(red_elephant.get_possible_moves(board))
    assert len(moves) == 2, "Red elephant should have 2 possible moves"


def test_horse_movement() -> None:
    """Test horse movement (check leg blockage)"""
    board = Board()

    red_horse = board.get_piece_at(Coordinate(1, 0))
    assert red_horse is not None and red_horse.piece_type == PieceType.HORSE
    moves = list(red_horse.get_possible_moves(board))
    assert len(moves) == 2, "Red horse should have 2 possible moves"


def test_coordinate_from_string() -> None:
    """Test Coordinate.from_string() conversion"""
    board = Board()

    valid_coords = [("e0", "Red king"), ("a9", "Black rook"), ("b0", "Red horse"), ("e5", "River")]
    for coord_str, _desc in valid_coords:
        coord = Coordinate.from_string(coord_str)
        assert coord.is_valid(), f"{coord_str} should be a valid coordinate"

    invalid_coords = ["j0", "a10", "E0", "e", "e0e", "00"]
    for coord_str in invalid_coords:
        try:
            Coordinate.from_string(coord_str)
            assert False, f"Expected ValueError for '{coord_str}'"
        except ValueError:
            pass


def test_is_move_legal() -> None:
    """Test direct move legality check"""
    board = Board()

    piece = board.get_piece_at(Coordinate(4, 0))
    assert piece is not None

    assert piece.is_move_legal(Coordinate(4, 0), Coordinate(4, 1), board), "King should be able to move forward one step"
    assert not piece.is_move_legal(Coordinate(4, 0), Coordinate(5, 0), board), "King should not be able to move sideways"


def test_generator_possible_moves() -> None:
    """Test generator for enumerating possible moves"""
    board = Board()

    piece = board.get_piece_at(Coordinate(4, 0))
    assert piece is not None

    moves_count = 0
    for _move in piece.get_possible_moves(board):
        moves_count += 1
    assert moves_count == 1, "King should have exactly 1 possible move"


def test_lazy_move_piece() -> None:
    """Verify move_piece uses is_move_legal internally"""
    board = Board()

    king_piece = board.get_piece_at(Coordinate(4, 0))
    assert king_piece is not None
    captured = board.move_piece(king_piece, Coordinate(4, 1))
    assert captured is None

    moved_king = board.get_piece_at(Coordinate(4, 1))
    assert moved_king is not None and moved_king.side == Side.RED


def test_illegal_move_fails() -> None:
    """Test that illegal moves raise ValueError"""
    board = Board()

    king_piece = board.get_piece_at(Coordinate(4, 0))
    assert king_piece is not None
    board.move_piece(king_piece, Coordinate(4, 1))

    try:
        king_piece_moved = board.get_piece_at(Coordinate(4, 1))
        assert king_piece_moved is not None
        board.move_piece(king_piece_moved, Coordinate(5, 2))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_get_pieces_by_side() -> None:
    """Test get_pieces_by_side returns correct pieces per side"""
    board = Board()

    red_pieces = list(board.get_pieces_by_side(Side.RED))
    black_pieces = list(board.get_pieces_by_side(Side.BLACK))

    assert len(red_pieces) == 16, f"Expected 16 red pieces, got {len(red_pieces)}"
    assert len(black_pieces) == 16, f"Expected 16 black pieces, got {len(black_pieces)}"

    for piece in red_pieces:
        assert piece.side == Side.RED, f"Expected RED piece, got {piece.side}"
    for piece in black_pieces:
        assert piece.side == Side.BLACK, f"Expected BLACK piece, got {piece.side}"

    from collections.abc import Iterator
    result = board.get_pieces_by_side(Side.RED)
    assert isinstance(result, Iterator), "Should return an iterator"


def test_move_piece_deepcopy_behavior() -> None:
    """Test move_piece object identity behavior for deepcopy flag"""
    board_shared = Board()
    shared_piece = board_shared.get_piece_at(Coordinate(4, 9))
    assert shared_piece is not None

    board_shared.move_piece(shared_piece, Coordinate(4, 8), deepcopy=False)
    moved_shared_piece = board_shared.get_piece_at(Coordinate(4, 8))
    assert moved_shared_piece is shared_piece, "deepcopy=False should keep same piece object"

    board_copied = Board()
    source_piece = board_copied.get_piece_at(Coordinate(4, 9))
    assert source_piece is not None

    board_copied.move_piece(source_piece, Coordinate(4, 8), deepcopy=True)
    moved_copied_piece = board_copied.get_piece_at(Coordinate(4, 8))
    assert moved_copied_piece is not None
    assert moved_copied_piece is not source_piece, "deepcopy=True should create new piece object"


def test_recover_restores_board_from_record() -> None:
    """Test recover can restore board state from MoveRecord"""
    board = Board()

    moving_piece = board.get_piece_at(Coordinate(1, 2))
    assert moving_piece is not None and moving_piece.piece_type == PieceType.CANNON
    captured_before = board.get_piece_at(Coordinate(1, 9))
    assert captured_before is not None and captured_before.piece_type == PieceType.HORSE

    from_pos = Coordinate(moving_piece.position.x, moving_piece.position.y)
    to_pos = Coordinate(1, 9)
    captured_piece = board.move_piece(moving_piece, to_pos)

    assert captured_piece is not None
    assert board.get_piece_at(from_pos) is None
    assert board.get_piece_at(to_pos) is moving_piece

    move_record = MoveRecord(from_pos=from_pos, to_pos=to_pos, captured=captured_piece)
    board.recover(move_record)

    restored_moving_piece = board.get_piece_at(from_pos)
    restored_captured_piece = board.get_piece_at(to_pos)
    assert restored_moving_piece is moving_piece, "Moved piece should return to original square"
    assert restored_captured_piece is captured_piece, "Captured piece should be restored"
    assert moving_piece.position == from_pos, "Moved piece position should be restored"


def test_board_equality() -> None:
    """Test Board.__eq__ for comparing board positions"""
    board1 = Board()
    board2 = Board()

    assert board1 == board2, "Two initial boards should be equal"

    assert board1 != "not a board", "Board should not equal a string"

    copied = board1.copy()
    assert board1 == copied, "Board.copy() should produce equal board"

    king = board1.get_piece_at(Coordinate(4, 0))
    assert king is not None
    board1.move_piece(king, Coordinate(4, 1))
    assert board1 != board2, "Boards after a move should differ"
    assert board1 != copied, "Board after move should differ from copy"

    king_moved = board1.get_piece_at(Coordinate(4, 1))
    assert king_moved is not None
    board1.move_piece(king_moved, Coordinate(4, 0))
    assert board1 == board2, "Restored board should equal original"


def run_all_tests() -> None:
    """Run all tests"""
    print("\n" + "=" * 50)
    print("Chinese Chess Class Test")
    print("=" * 50 + "\n")

    tests = [
        ("Initial Board", test_initial_board),
        ("King Movement", test_king_movement),
        ("Advisor Movement", test_advisor_movement),
        ("Chariot Movement", test_chariot_movement),
        ("Piece Movement Execution", test_piece_movement),
        ("Elephant Movement", test_elephant_movement),
        ("Horse Movement", test_horse_movement),
        ("Coordinate String Conversion", test_coordinate_from_string),
        ("Direct Move Legality Check", test_is_move_legal),
        ("Generator Possible Moves", test_generator_possible_moves),
        ("Lazy Move Piece", test_lazy_move_piece),
        ("Illegal Move", test_illegal_move_fails),
        ("Get Pieces By Side", test_get_pieces_by_side),
        ("Move Piece Deepcopy Behavior", test_move_piece_deepcopy_behavior),
        ("Recover Board From MoveRecord", test_recover_restores_board_from_record),
        ("Board Equality", test_board_equality),
    ]

    failures = 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"✓ {name} passed\n")
        except (AssertionError, Exception) as e:
            print(f"✗ {name} failed: {e}\n")
            failures += 1

    print("=" * 50)
    if failures == 0:
        print("All tests passed! ✓")
    else:
        print(f"{failures} test(s) failed! ✗")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_all_tests()
