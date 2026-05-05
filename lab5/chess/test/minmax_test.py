from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.minmax as minmax
from src.chess import Coordinate, Piece, PieceType, Side, Board


def _assert_best_move(
    pieces: list[Piece],
    turn: Side,
    expected: tuple[Coordinate, Coordinate],
    case_name: str,
) -> None:
    board: Board = Board.from_pieces(pieces)
    result = minmax.solve(board, turn)
    assert result == expected, f"{case_name} failed: Expected {expected}, but got {result}"


def one_step_test() -> None:
    _assert_best_move(
        pieces=[
            Piece(PieceType.KING, Side.RED, Coordinate(5, 0)),
            Piece(PieceType.KING, Side.BLACK, Coordinate(4, 9)),
            Piece(PieceType.CHARIOT, Side.BLACK, Coordinate(5, 3)),
        ],
        turn=Side.BLACK,
        expected=(Coordinate(5, 3), Coordinate(5, 0)),
        case_name="Case 1",
    )

    _assert_best_move(
        pieces=[
            Piece(PieceType.KING, Side.RED, Coordinate(5, 0)),
            Piece(PieceType.KING, Side.BLACK, Coordinate(4, 9)),
            Piece(PieceType.CHARIOT, Side.BLACK, Coordinate(4, 3)),
            Piece(PieceType.CHARIOT, Side.BLACK, Coordinate(5, 5)),
        ],
        turn=Side.BLACK,
        expected=(Coordinate(5, 5), Coordinate(5, 0)),
        case_name="Case 2",
    )

def checked_test() -> None:
    _assert_best_move(
        pieces=[
            Piece(PieceType.KING, Side.RED, Coordinate(5, 0)),
            Piece(PieceType.CHARIOT, Side.RED, Coordinate(1, 5)),
            Piece(PieceType.KING, Side.BLACK, Coordinate(3, 9)),
            Piece(PieceType.CHARIOT, Side.BLACK, Coordinate(5, 3)),
            Piece(PieceType.HORSE, Side.BLACK, Coordinate(7, 2)),
        ],
        turn=Side.RED,
        expected=(Coordinate(5, 0), Coordinate(4, 0)),
        case_name="Checked Case",
    )

if __name__ == "__main__":
    one_step_test()
    checked_test()
    print("✓ All minmax tests passed")
