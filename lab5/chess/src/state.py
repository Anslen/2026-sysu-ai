"""Game state management for minimax search.

Stores board state, current player, last move record, and the
single piece captured on the last move (if any).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, cast
from copy import deepcopy

from src.chess import Board, Coordinate, Piece, PieceType, Side

PieceValue: dict[PieceType, int] = {
    PieceType.ADVISOR: 200,
    PieceType.ELEPHANT: 200,
    PieceType.HORSE: 450,
    PieceType.CHARIOT: 900,
    PieceType.CANNON: 650,
    PieceType.SOLDIER: 30,
}

SEARCH_DEPTH: int = 6


@dataclass
class MoveRecord:
    """Record of a single move (from, to, and optionally a captured piece)."""
    __slots__ = ('from_pos', 'to_pos', 'captured')
    from_pos: Coordinate
    to_pos: Coordinate
    captured: Optional[Piece]


@dataclass
class GameState:
    """Complete game state snapshot for minimax search.

    The board is shallow-copied; Piece objects are shared between states.
    The captured piece on the last move always belongs to current_turn,
    and at most one piece can be captured per move.

    Attributes:
        board: Current board layout.
        current_turn: Side to move next (RED / BLACK).
        last_move: Move record of the previous move (None on first turn).
        last_captured: Piece captured on the last move (None if no capture).
                       Always belongs to current_turn.
    """
    __slots__ = (
        "board",
        "current_turn",
        "score",

        "left_depth",

        "child",

        "piece_iter",
        "checking_piece",
        "move_iter",
        "has_vaild_move",

        "last_move",
        "last_captured",

        "alpha",
        "beta",
        "result_left_depth",

        "red_king_captured",
        "black_king_captured",
    )
    board          : Board
    current_turn   : Side
    score          : int


    left_depth     : int
    child          : Optional[GameState]

    piece_iter     : Iterator[Piece]
    checking_piece : Optional[Piece]
    move_iter      : Optional[Iterator[Coordinate]]
    has_vaild_move : bool

    last_move      : Optional[MoveRecord]
    last_captured  : Optional[Piece]

    alpha          : int
    beta           : int
    result_left_depth : int

    red_king_captured   : bool
    black_king_captured : bool

    @classmethod
    def from_board(cls, board: Board, current_turn: Side = Side.RED) -> GameState:
        """Create an initial GameState from a board.

        Args:
            board: The initial board layout.
            current_turn: The side that moves first (default RED).

        Returns:
            A GameState with no move history and no captured piece.
        """
        return cls(
            board          = board,
            current_turn   = current_turn,
            score          = 0,
            left_depth     = SEARCH_DEPTH,
            child          = None,
            piece_iter     = board.get_pieces_by_side(current_turn),
            checking_piece = None,
            move_iter      = None,
            has_vaild_move = False,
            last_move      = None,
            last_captured  = None,
            alpha          = -100000,
            beta           = 100000,
            result_left_depth = -1,
            red_king_captured   = False,
            black_king_captured = False,
        )

    def apply_move(self, piece: Piece, to_pos: Coordinate, deep_copy: bool = False) -> GameState:
        """Apply a move and return a new GameState without mutating self.

        Because the board stores only shallow copies of Piece objects,
        a *new* Piece is created for the moved piece so the original
        board is never altered.

        Args:
            piece: The piece to move.
            to_pos: Target position.

        Returns:
            A new GameState reflecting the move.

        Raises:
            KeyError: No piece at from_pos.
        """
        # Shallow copy — same Piece objects, independent dict
        new_board: Board
        if deep_copy:
            new_board = deepcopy(self.board)
        else:
            new_board = self.board

        # Check for an enemy piece at the target (capture)
        captured: Optional[Piece] = new_board.move_piece(piece, to_pos, deepcopy=True)

        # Handle piece captured
        new_score: int = self.score
        red_king_captured: bool = self.red_king_captured
        black_king_captured: bool = self.black_king_captured
        if captured is not None:
            # Track king captures via flags, update score only for non-king pieces
            if captured.piece_type == PieceType.KING:
                if captured.side == Side.RED:
                    red_king_captured = True
                else:
                    black_king_captured = True
            else:
                captured_value: int = PieceValue[captured.piece_type]
                if self.current_turn == Side.RED:
                    new_score += captured_value
                else:
                    new_score -= captured_value

        # Build the move record
        move_record: MoveRecord = MoveRecord(
            piece.position,
            to_pos,
            captured,
        )

        # Switch turns
        next_turn: Side = Side.BLACK if self.current_turn == Side.RED else Side.RED

        # Pass alpha-beta to child
        alpha: int = -100000
        beta: int = 100000
        if self.current_turn == Side.RED:
            alpha = self.alpha
        else:
            beta = self.beta

        return GameState(
            board          = new_board,
            current_turn   = next_turn,
            score          = new_score,
            left_depth     = self.left_depth - 1,
            child          = None,
            piece_iter     = new_board.get_pieces_by_side(next_turn),
            checking_piece = None,
            move_iter      = None,
            has_vaild_move = False,
            last_move      = move_record,
            last_captured  = captured,
            alpha          = alpha,
            beta           = beta,
            result_left_depth = -1,
            red_king_captured   = red_king_captured,
            black_king_captured = black_king_captured,
        )

    def recover(self):
        """Recover the board state by undoing the last move."""
        self.board.recover(cast(MoveRecord, self.last_move))
