from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from src.state import MoveRecord

# Movement direction constants (module-level to avoid repeated allocation)
_KING_DIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))
_ADVISOR_DIRS: tuple[tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_ELEPHANT_DIRS: tuple[tuple[int, int], ...] = ((2, 2), (2, -2), (-2, 2), (-2, -2))
_HORSE_MOVES: tuple[tuple[int, int], ...] = (
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
)
_ORTHOGONAL_DIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))


class PieceType(Enum):
    """Piece type enumeration"""
    KING = "帅/将"  # King (red/black)
    ADVISOR = "士/仕"  # Advisor (red/black)
    ELEPHANT = "象/相"  # Elephant (red/black)
    HORSE = "马"
    CHARIOT = "车"
    CANNON = "炮"
    SOLDIER = "兵/卒"  # Soldier (red/black)


class Side(Enum):
    """Player side (Red/Black)"""
    RED = "红"
    BLACK = "黑"


@dataclass
class Coordinate:
    """Board coordinate (x: 0-8 columns, y: 0-9 rows).
    
    Internal storage: x(0-8), y(0-9) where y=0 is bottom (red side), y=9 is top (black side).
    Display format: col(a-i), row(0-9) where row=0 is bottom, row=9 is top.
    """
    __slots__ = ('x', 'y')
    x: int  # column (0-8, displayed as a-i)
    y: int  # row (0-9, displayed as 0-9 from bottom to top)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Coordinate):
            return False
        return self.x == other.x and self.y == other.y

    def __repr__(self) -> str:
        """Display in modern coordinate format (a-i columns, 0-9 rows, left-bottom origin)"""
        col: str = chr(ord('a') + self.x)
        row: int = self.y
        return f"{col}{row}"

    @staticmethod
    def from_string(coord_str: str) -> Coordinate:
        """Convert string format (e.g., 'e0', 'a9') to Coordinate.
        
        Args:
            coord_str: String in format 'col+row' where col is a-i and row is 0-9
            
        Returns:
            Coordinate object with internal representation (x, y)
            
        Raises:
            ValueError: If string format is invalid or coordinates are out of bounds
        """
        if not isinstance(coord_str, str) or len(coord_str) != 2:
            raise ValueError(f"Invalid coordinate format: '{coord_str}'. Expected format: 'a0' to 'i9'")
        
        col_char: str = coord_str[0]
        row_char: str = coord_str[1]
        
        # Validate column (a-i)
        if col_char < 'a' or col_char > 'i':
            raise ValueError(f"Invalid column: '{col_char}'. Expected a-i")
        
        # Validate row (0-9)
        if not row_char.isdigit() or int(row_char) < 0 or int(row_char) > 9:
            raise ValueError(f"Invalid row: '{row_char}'. Expected 0-9")
        
        # Convert to internal coordinates
        x: int = ord(col_char) - ord('a')
        y: int = int(row_char)
        
        return Coordinate(x, y)

    def is_valid(self) -> bool:
        """Check if coordinate is within board boundaries"""
        return 0 <= self.x <= 8 and 0 <= self.y <= 9

    def is_in_palace(self, side: Side) -> bool:
        """Check if coordinate is within palace (red palace: y=0-2, black palace: y=7-9)"""
        if side == Side.RED:
            return 3 <= self.x <= 5 and 0 <= self.y <= 2
        else:
            return 3 <= self.x <= 5 and 7 <= self.y <= 9


@dataclass
class Piece:
    """Piece class"""
    __slots__ = ('piece_type', 'side', 'position')
    piece_type: PieceType
    side: Side
    position: Coordinate

    def __repr__(self) -> str:
        return f"{self.side.value}{self.piece_type.value}@{self.position}"

    def is_move_legal(self, from_pos: Coordinate, to_pos: Coordinate, board: Board) -> bool:
        """Check if a move from from_pos to to_pos is legal.
        
        Args:
            from_pos: Starting position
            to_pos: Target position
            board: Current board state
            
        Returns:
            True if move is legal, False otherwise
        """
        if from_pos != self.position:
            return False
        
        if self.piece_type == PieceType.KING:
            return self._check_king_move(to_pos, board)
        elif self.piece_type == PieceType.ADVISOR:
            return self._check_advisor_move(to_pos, board)
        elif self.piece_type == PieceType.ELEPHANT:
            return self._check_elephant_move(to_pos, board)
        elif self.piece_type == PieceType.HORSE:
            return self._check_horse_move(to_pos, board)
        elif self.piece_type == PieceType.CHARIOT:
            return self._check_chariot_move(to_pos, board)
        elif self.piece_type == PieceType.CANNON:
            return self._check_cannon_move(to_pos, board)
        elif self.piece_type == PieceType.SOLDIER:
            return self._check_soldier_move(to_pos, board)
        return False

    def get_possible_moves(self, board: Board) -> Iterator[Coordinate]:
        """Generate all possible move positions (iterator).
        
        Yields:
            Coordinate: Each valid move position
        """
        # Use piece-specific movement logic instead of brute-force enumeration
        if self.piece_type == PieceType.KING:
            yield from self._generate_king_moves(board)
        elif self.piece_type == PieceType.ADVISOR:
            yield from self._generate_advisor_moves(board)
        elif self.piece_type == PieceType.ELEPHANT:
            yield from self._generate_elephant_moves(board)
        elif self.piece_type == PieceType.HORSE:
            yield from self._generate_horse_moves(board)
        elif self.piece_type == PieceType.CHARIOT:
            yield from self._generate_chariot_moves(board)
        elif self.piece_type == PieceType.CANNON:
            yield from self._generate_cannon_moves(board)
        elif self.piece_type == PieceType.SOLDIER:
            yield from self._generate_soldier_moves(board)

    def _check_king_move(self, to_pos: Coordinate, board: Board) -> bool:
        """Check if king/general can move one step within palace, or fly to capture opposing king"""
        dx = to_pos.x - self.position.x
        dy = to_pos.y - self.position.y
        adx = abs(dx)
        ady = abs(dy)
        # King moves exactly one step within palace
        if (adx == 1 and ady == 0) or (adx == 0 and ady == 1):
            return to_pos.is_in_palace(self.side) and board.can_move_to(to_pos, self.side)
        # Flying general: same column, target is opponent's king, path is clear
        if dx == 0 and ady > 1:
            target = board.get_piece_at(to_pos)
            if target is not None and target.piece_type == PieceType.KING and target.side != self.side:
                step_y = 1 if dy > 0 else -1
                check_y = self.position.y + step_y
                while check_y != to_pos.y:
                    if not board.is_empty(Coordinate(self.position.x, check_y)):
                        return False
                    check_y += step_y
                return True
        return False

    def _generate_king_moves(self, board: Board) -> Iterator[Coordinate]:
        """Generate all legal king moves: one-step within palace, plus flying general capture"""
        # Normal one-step moves within palace
        for dx, dy in _KING_DIRS:
            new_pos = Coordinate(self.position.x + dx, self.position.y + dy)
            if new_pos.is_valid() and new_pos.is_in_palace(self.side):
                if board.can_move_to(new_pos, self.side):
                    yield new_pos
        # Flying general: scan column for opponent's king
        opponent_side = Side.BLACK if self.side == Side.RED else Side.RED
        for direction in (-1, 1):
            check_y = self.position.y + direction
            while 0 <= check_y <= 9:
                check_pos = Coordinate(self.position.x, check_y)
                piece = board.get_piece_at(check_pos)
                if piece is not None:
                    if piece.piece_type == PieceType.KING and piece.side == opponent_side:
                        yield check_pos
                    break
                check_y += direction

    def _check_advisor_move(self, to_pos: Coordinate, board: Board) -> bool:
        """Check if advisor can move one step diagonally within palace"""
        dx = abs(to_pos.x - self.position.x)
        dy = abs(to_pos.y - self.position.y)
        # Advisor moves exactly one step diagonally
        if dx == 1 and dy == 1:
            return to_pos.is_in_palace(self.side) and board.can_move_to(to_pos, self.side)
        return False

    def _generate_advisor_moves(self, board: Board) -> Iterator[Coordinate]:
        """Generate all legal advisor moves using directional constants"""
        for dx, dy in _ADVISOR_DIRS:
            new_pos = Coordinate(self.position.x + dx, self.position.y + dy)
            if new_pos.is_valid() and new_pos.is_in_palace(self.side):
                if board.can_move_to(new_pos, self.side):
                    yield new_pos

    def _check_elephant_move(self, to_pos: Coordinate, board: Board) -> bool:
        """Check if elephant can move two steps diagonally without crossing river"""
        dx = to_pos.x - self.position.x
        dy = to_pos.y - self.position.y
        
        # Elephant moves exactly 2 steps diagonally
        if abs(dx) != 2 or abs(dy) != 2:
            return False
        
        # Check if crossing river (river is at y=4-5)
        if self.side == Side.RED and to_pos.y > 4:
            return False
        if self.side == Side.BLACK and to_pos.y < 5:
            return False
        
        # Check if elephant eye (middle position) is blocked
        eye_x = self.position.x + dx // 2
        eye_y = self.position.y + dy // 2
        eye_pos = Coordinate(eye_x, eye_y)
        
        return board.is_empty(eye_pos) and board.can_move_to(to_pos, self.side)

    def _generate_elephant_moves(self, board: Board) -> Iterator[Coordinate]:
        """Generate all legal elephant moves considering river and eye blocking"""
        for dx, dy in _ELEPHANT_DIRS:
            new_pos = Coordinate(self.position.x + dx, self.position.y + dy)
            
            if not new_pos.is_valid():
                continue
            
            # Check if crossing river (river is at y=4-5)
            if self.side == Side.RED and new_pos.y > 4:
                continue
            if self.side == Side.BLACK and new_pos.y < 5:
                continue
            
            # Check if elephant eye (middle position) is blocked
            eye_x = self.position.x + dx // 2
            eye_y = self.position.y + dy // 2
            eye_pos = Coordinate(eye_x, eye_y)
            
            if board.is_empty(eye_pos) and board.can_move_to(new_pos, self.side):
                yield new_pos

    def _check_horse_move(self, to_pos: Coordinate, board: Board) -> bool:
        """Check if horse can move in L-shape with leg not blocked"""
        dx = to_pos.x - self.position.x
        dy = to_pos.y - self.position.y
        
        # Horse moves in L-shape: 2 in one direction, 1 perpendicular
        l_shape = (abs(dx) == 2 and abs(dy) == 1) or (abs(dx) == 1 and abs(dy) == 2)
        if not l_shape:
            return False
        
        # Check if horse's leg is blocked (one step in the 2-step direction)
        if abs(dx) == 2:
            leg_x = self.position.x + dx // 2
            leg_y = self.position.y
        else:
            leg_x = self.position.x
            leg_y = self.position.y + dy // 2
        leg_pos = Coordinate(leg_x, leg_y)
        
        return board.is_empty(leg_pos) and board.can_move_to(to_pos, self.side)

    def _generate_horse_moves(self, board: Board) -> Iterator[Coordinate]:
        """Generate all legal horse moves considering leg blocking"""
        for dx, dy in _HORSE_MOVES:
            new_pos = Coordinate(self.position.x + dx, self.position.y + dy)
            
            if not new_pos.is_valid():
                continue
            
            # Check if horse's leg is blocked (one step in the 2-step direction)
            if abs(dx) == 2:
                leg_x = self.position.x + dx // 2
                leg_y = self.position.y
            else:
                leg_x = self.position.x
                leg_y = self.position.y + dy // 2
            leg_pos = Coordinate(leg_x, leg_y)
            
            if board.is_empty(leg_pos) and board.can_move_to(new_pos, self.side):
                yield new_pos

    def _check_chariot_move(self, to_pos: Coordinate, board: Board) -> bool:
        """Check if chariot can move horizontally or vertically without jumping"""
        dx = to_pos.x - self.position.x
        dy = to_pos.y - self.position.y
        
        # Chariot moves horizontally or vertically
        if dx != 0 and dy != 0:
            return False
        
        if dx == 0 and dy == 0:
            return False
        
        # Check path is clear
        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0
        
        current_x = self.position.x + step_x
        current_y = self.position.y + step_y
        
        while current_x != to_pos.x or current_y != to_pos.y:
            if not board.is_empty(Coordinate(current_x, current_y)):
                return False
            current_x += step_x
            current_y += step_y
        
        return board.can_move_to(to_pos, self.side)

    def _generate_chariot_moves(self, board: Board) -> Iterator[Coordinate]:
        """Generate all legal chariot moves along four orthogonal directions"""
        for dx, dy in _ORTHOGONAL_DIRS:
            step = 1
            while True:
                new_pos = Coordinate(self.position.x + dx * step, self.position.y + dy * step)
                
                if not new_pos.is_valid():
                    break
                
                if board.is_empty(new_pos):
                    yield new_pos
                else:
                    # If it's an opponent's piece, can capture it
                    if board.can_move_to(new_pos, self.side):
                        yield new_pos
                    break
                
                step += 1

    def _check_cannon_move(self, to_pos: Coordinate, board: Board) -> bool:
        """Check if cannon can move like chariot or capture by jumping"""
        dx = to_pos.x - self.position.x
        dy = to_pos.y - self.position.y
        
        # Cannon moves horizontally or vertically
        if dx != 0 and dy != 0:
            return False
        
        if dx == 0 and dy == 0:
            return False
        
        # Count pieces in path
        step_x = 1 if dx > 0 else -1 if dx < 0 else 0
        step_y = 1 if dy > 0 else -1 if dy < 0 else 0
        
        current_x = self.position.x + step_x
        current_y = self.position.y + step_y
        piece_count = 0
        
        while current_x != to_pos.x or current_y != to_pos.y:
            if not board.is_empty(Coordinate(current_x, current_y)):
                piece_count += 1
                if piece_count > 1:
                    return False
            current_x += step_x
            current_y += step_y
        
        # If no pieces jumped, can move normally
        if piece_count == 0:
            return board.can_move_to(to_pos, self.side)
        # If one piece jumped, can capture
        else:
            return not board.is_empty(to_pos) and board.can_move_to(to_pos, self.side)

    def _generate_cannon_moves(self, board: Board) -> Iterator[Coordinate]:
        """Generate all legal cannon moves (same path as chariot, different capture rule)"""
        for dx, dy in _ORTHOGONAL_DIRS:
            step = 1
            jumped = False
            
            while True:
                new_pos = Coordinate(self.position.x + dx * step, self.position.y + dy * step)
                
                if not new_pos.is_valid():
                    break
                
                if board.is_empty(new_pos):
                    # If haven't jumped over a piece yet, can move directly
                    if not jumped:
                        yield new_pos
                else:
                    # Encountered a piece
                    if not jumped:
                        # First encounter, mark as jumped and continue looking for capture target
                        jumped = True
                    else:
                        # Already jumped over one, this is the second, can capture if opponent's
                        if board.can_move_to(new_pos, self.side):
                            yield new_pos
                        break
                
                step += 1

    def _check_soldier_move(self, to_pos: Coordinate, board: Board) -> bool:
        """Check if soldier can move according to crossing river status"""
        dx = abs(to_pos.x - self.position.x)
        dy = abs(to_pos.y - self.position.y)
        
        # Must move exactly one step
        if (dx + dy) != 1:
            return False
        
        # Determine forward direction and crossing status
        if self.side == Side.RED:
            forward_dy = 1
            has_crossed = self.position.y >= 5
        else:
            forward_dy = -1
            has_crossed = self.position.y <= 4
        
        actual_dy = to_pos.y - self.position.y
        
        if not has_crossed:
            # Not crossed river, can only move forward
            return actual_dy == forward_dy and board.can_move_to(to_pos, self.side)
        else:
            # Crossed river, can move forward, left, or right
            return (actual_dy == forward_dy or actual_dy == 0) and board.can_move_to(to_pos, self.side)

    def _generate_soldier_moves(self, board: Board) -> Iterator[Coordinate]:
        """Generate all legal soldier moves based on river crossing status"""
        if self.side == Side.RED:
            forward_dy = 1
            has_crossed = self.position.y >= 5
        else:
            forward_dy = -1
            has_crossed = self.position.y <= 4
        
        if not has_crossed:
            # Not crossed river, can only move forward
            new_pos = Coordinate(self.position.x, self.position.y + forward_dy)
            if new_pos.is_valid() and board.can_move_to(new_pos, self.side):
                yield new_pos
        else:
            # Crossed river, can move forward, left, or right
            for dx, dy in ((0, forward_dy), (-1, 0), (1, 0)):
                new_pos = Coordinate(self.position.x + dx, self.position.y + dy)
                if new_pos.is_valid() and board.can_move_to(new_pos, self.side):
                    yield new_pos


class Board:
    """Chinese Chess Board"""
    __slots__ = ('_pieces',)
    
    WIDTH: int = 9  # number of columns
    HEIGHT: int = 10  # number of rows
    RIVER_Y: int = 5  # river boundary (between y=4 and y=5)
    
    # Display names for each piece type (red, black)
    _PIECE_DISPLAY_NAMES: dict[PieceType, tuple[str, str]] = {
        PieceType.KING: ("帅", "将"),
        PieceType.ADVISOR: ("士", "仕"),
        PieceType.ELEPHANT: ("象", "相"),
        PieceType.HORSE: ("马", "马"),
        PieceType.CHARIOT: ("车", "车"),
        PieceType.CANNON: ("炮", "炮"),
        PieceType.SOLDIER: ("兵", "卒"),
    }
    
    _pieces: dict[Coordinate, Piece]

    @classmethod
    def from_pieces(cls, pieces: list[Piece]) -> Board:
        ret: Board = cls.__new__(cls)
        ret._pieces = {piece.position: piece for piece in pieces}
        return ret

    def __init__(self) -> None:
        """Initialize board"""
        # Store pieces in dictionary, key is coordinate, value is piece
        self._pieces = {}
        self._init_default_setup()
    
    def _init_default_setup(self) -> None:
        """Initialize default piece layout"""
        # Red layout (y = 0-2)
        # Back row (y=0): chariot, horse, elephant, advisor, king, advisor, elephant, horse, chariot
        self._add_piece(Piece(PieceType.CHARIOT, Side.RED, Coordinate(0, 0)))
        self._add_piece(Piece(PieceType.HORSE, Side.RED, Coordinate(1, 0)))
        self._add_piece(Piece(PieceType.ELEPHANT, Side.RED, Coordinate(2, 0)))
        self._add_piece(Piece(PieceType.ADVISOR, Side.RED, Coordinate(3, 0)))
        self._add_piece(Piece(PieceType.KING, Side.RED, Coordinate(4, 0)))
        self._add_piece(Piece(PieceType.ADVISOR, Side.RED, Coordinate(5, 0)))
        self._add_piece(Piece(PieceType.ELEPHANT, Side.RED, Coordinate(6, 0)))
        self._add_piece(Piece(PieceType.HORSE, Side.RED, Coordinate(7, 0)))
        self._add_piece(Piece(PieceType.CHARIOT, Side.RED, Coordinate(8, 0)))
        
        # Cannons (y=2): cannon _ _ _ _ _ _ _ cannon
        self._add_piece(Piece(PieceType.CANNON, Side.RED, Coordinate(1, 2)))
        self._add_piece(Piece(PieceType.CANNON, Side.RED, Coordinate(7, 2)))
        
        # Soldiers (y=3): soldier _ soldier _ soldier _ soldier _ soldier
        self._add_piece(Piece(PieceType.SOLDIER, Side.RED, Coordinate(0, 3)))
        self._add_piece(Piece(PieceType.SOLDIER, Side.RED, Coordinate(2, 3)))
        self._add_piece(Piece(PieceType.SOLDIER, Side.RED, Coordinate(4, 3)))
        self._add_piece(Piece(PieceType.SOLDIER, Side.RED, Coordinate(6, 3)))
        self._add_piece(Piece(PieceType.SOLDIER, Side.RED, Coordinate(8, 3)))
        
        # Black layout (y = 7-9)
        # Back row (y=9): chariot, horse, elephant, advisor, general, advisor, elephant, horse, chariot
        self._add_piece(Piece(PieceType.CHARIOT, Side.BLACK, Coordinate(0, 9)))
        self._add_piece(Piece(PieceType.HORSE, Side.BLACK, Coordinate(1, 9)))
        self._add_piece(Piece(PieceType.ELEPHANT, Side.BLACK, Coordinate(2, 9)))
        self._add_piece(Piece(PieceType.ADVISOR, Side.BLACK, Coordinate(3, 9)))
        self._add_piece(Piece(PieceType.KING, Side.BLACK, Coordinate(4, 9)))
        self._add_piece(Piece(PieceType.ADVISOR, Side.BLACK, Coordinate(5, 9)))
        self._add_piece(Piece(PieceType.ELEPHANT, Side.BLACK, Coordinate(6, 9)))
        self._add_piece(Piece(PieceType.HORSE, Side.BLACK, Coordinate(7, 9)))
        self._add_piece(Piece(PieceType.CHARIOT, Side.BLACK, Coordinate(8, 9)))
        
        # Cannons (y=7): cannon _ _ _ _ _ _ _ cannon
        self._add_piece(Piece(PieceType.CANNON, Side.BLACK, Coordinate(1, 7)))
        self._add_piece(Piece(PieceType.CANNON, Side.BLACK, Coordinate(7, 7)))
        
        # Soldiers (y=6): soldier _ soldier _ soldier _ soldier _ soldier
        self._add_piece(Piece(PieceType.SOLDIER, Side.BLACK, Coordinate(0, 6)))
        self._add_piece(Piece(PieceType.SOLDIER, Side.BLACK, Coordinate(2, 6)))
        self._add_piece(Piece(PieceType.SOLDIER, Side.BLACK, Coordinate(4, 6)))
        self._add_piece(Piece(PieceType.SOLDIER, Side.BLACK, Coordinate(6, 6)))
        self._add_piece(Piece(PieceType.SOLDIER, Side.BLACK, Coordinate(8, 6)))
    
    def _add_piece(self, piece: Piece) -> None:
        """Add a piece to the board"""
        self._pieces[piece.position] = piece
    
    def __eq__(self, other: object) -> bool:
        """Check if two boards have the same position (same pieces at same coordinates)."""
        if not isinstance(other, Board):
            return False
        return self._pieces == other._pieces

    def get_piece_at(self, position: Coordinate) -> Optional[Piece]:
        """Get piece at specified position"""
        return self._pieces.get(position)
    
    def is_empty(self, position: Coordinate) -> bool:
        """Check if position is empty"""
        return position not in self._pieces
    
    def can_move_to(self, position: Coordinate, side: Side) -> bool:
        """Check if can move to position (position is empty or has opponent's piece)"""
        piece: Optional[Piece] = self._pieces.get(position)
        return piece is None or piece.side != side
    
    def get_pieces_by_side(self, side: Side) -> Iterator[Piece]:
        """Return an iterator over all pieces belonging to the given side.
        
        Args:
            side: RED or BLACK
            
        Yields:
            Piece: Each piece belonging to the specified side
        """
        return iter(tuple(filter(lambda x: x.side == side, self._pieces.values())))
    
    def move_piece(self, piece: Piece, to_pos: Coordinate, deepcopy: bool = False) -> Optional[Piece]:
        """Move a piece from from_pos to to_pos.
        
        Args:
            from_pos: Starting position
            to_pos: Target position
            deepcopy: Whether to perform a deep copy of the board
            
        Returns:
            Captured piece if there was one, None if the move was to an empty square
            
        Raises:
            ValueError: If from_pos is empty or the move is not legal
        """
        from_pos: Coordinate = piece.position

        # Check if move is legal using the piece's check method
        if not piece.is_move_legal(from_pos, to_pos, self):
            raise ValueError(f"Illegal move: {piece} cannot move to {to_pos}")
        
        # Get captured piece if any
        captured_piece: Optional[Piece] = self.get_piece_at(to_pos)
        
        # Execute move
        del self._pieces[from_pos]
        if deepcopy:
            new_piece: Piece = Piece(piece.piece_type, piece.side, to_pos)
            self._pieces[to_pos] = new_piece
        else:
            piece.position = to_pos
            self._pieces[to_pos] = piece
        
        return captured_piece
    
    def copy(self) -> Board:
        """Create a shallow copy of the board (shared Piece objects, new dict).
        
        Returns:
            A new Board referencing the same Piece objects.
        """
        new_board: Board = Board.__new__(Board)
        new_board._pieces = dict(self._pieces)
        return new_board

    def recover(self, move_record: MoveRecord):
        """Recover the board state by undoing the last move."""
        piece: Piece = self._pieces.pop(move_record.to_pos)
        piece.position = move_record.from_pos
        self._pieces[move_record.from_pos] = piece
        if move_record.captured is not None:
            self._pieces[move_record.to_pos] = move_record.captured

    def _get_display_width(self, text: str) -> int:
        """Calculate display width considering Chinese characters take 2 widths"""
        width: int = 0
        for char in text:
            # Chinese characters (CJK Unified Ideographs) and special symbols take 2 display widths
            if '\u4e00' <= char <= '\u9fff' or char == '〜':
                width += 2
            else:
                width += 1
        return width
    
    def _pad_cell(self, text: str, target_width: int) -> str:
        """Pad text to target display width with spaces"""
        current_width: int = self._get_display_width(text)
        padding_needed: int = target_width - current_width
        if padding_needed < 0:
            padding_needed = 0
        return text + " " * padding_needed
    
    def __repr__(self) -> str:
        """Display board with modern coordinates (a-i columns, 0-9 rows, left-bottom origin)"""
        board_str: str = ""
        cell_width: int = 6  # Display width for each cell
        border_char: str = "+"
        h_line_char: str = "-"
        
        # Pre-calculate border line
        border_line = "  " + border_char + h_line_char * (cell_width * self.WIDTH + 1) + border_char + "\n"
        board_str += border_line
        
        # Display rows from y=9 to y=0 (which represent rows 9 to 0 in modern notation)
        for y in range(self.HEIGHT - 1, -1, -1):
            # Modern row number equals y directly
            modern_row: int = y
            board_str += f"{modern_row:2d} | "
            
            for x in range(self.WIDTH):
                piece: Optional[Piece] = self._pieces.get(Coordinate(x, y))
                
                if piece is None:
                    # Empty cell
                    cell_text = "〜" if y == 4 or y == 5 else "·"
                else:
                    # Display piece with side indicator
                    side_prefix = "红" if piece.side == Side.RED else "黑"
                    piece_names = self._PIECE_DISPLAY_NAMES[piece.piece_type]
                    piece_name = piece_names[0] if piece.side == Side.RED else piece_names[1]
                    cell_text = f"{side_prefix}{piece_name}"
                
                # Pad to cell width
                padded_text = self._pad_cell(cell_text, cell_width)
                board_str += padded_text
            
            board_str += "|\n"
        
        # Bottom border
        board_str += border_line
        
        # Column header (a-i) at bottom
        board_str += "     "  # Space for row numbers and left border
        for x in range(self.WIDTH):
            # Column letter (a-i)
            col_str = chr(ord('a') + x)
            padded_col = self._pad_cell(col_str, cell_width)
            board_str += padded_col
        board_str += "\n"
        
        return board_str


if __name__ == "__main__":
    # Example usage
    board: Board = Board()
    print("Initial board:")
    print(board)
    
    # Test getting possible moves of a piece
    red_king: Optional[Piece] = board.get_piece_at(Coordinate(4, 9))
    if red_king is not None:
        print(f"\nPossible moves for {red_king}:")
        possible_moves: Iterator[Coordinate] = red_king.get_possible_moves(board)
        for move in possible_moves:
            print(f"  {move}")
    
    # Test moving a piece
    red_soldier: Optional[Piece] = board.get_piece_at(Coordinate(1, 6))
    if red_soldier is not None:
        print(f"\nPossible moves for {red_soldier}:")
        possible_moves: Iterator[Coordinate] = red_soldier.get_possible_moves(board)
        for move in possible_moves:
            print(f"  {move}")
