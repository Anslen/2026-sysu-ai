# AGENTS.md — Chinese Chess (Xiangqi) Engine

## Project Overview

A Chinese Chess (Xiangqi) game engine implemented in pure Python (no external dependencies). All core logic lives in a single ~660-line file. The project demonstrates Python optimization techniques: `__slots__`, module-level constants, generator-based lazy evaluation, and a dual-API approach for move validation.

---

## Essential Commands

No build system exists. All commands are run directly with `python`.

```powershell
# Run all tests (12 tests)
python test/chess_test.py

# Run main module directly
python src/chess.py
```

**Note**: Always run commands from `C:/code/python/homework/ai/lab5/chess/`. Do NOT `cd` into `src/` or `test/` — the import paths are relative to the project root.

---

## Code Structure

```
chess/
├── src/
│   ├── __init__.py          # Empty package marker
│   └── chess.py             # All core classes (664 lines)
├── test/
│   ├── __init__.py          # Empty package marker
│   └── chess_test.py        # All 12 test functions (no framework)
├── docs/
│   ├── README.md            # Chinese-language project README
│   ├── PROJECT_STRUCTURE.md # File layout (Chinese)
│   ├── OPTIMIZATION.md      # Memory optimization documentation
│   └── LAZY_EVALUATION.md   # Lazy evaluation design doc
└── AGENTS.md                # This file
```

---

## Architecture & Data Flow

### Core Classes (all in `src/chess.py`)

| Class | Description | Slots |
|-------|-------------|-------|
| `PieceType(Enum)` | 7 piece types: KING, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER | — |
| `Side(Enum)` | RED/BLACK | — |
| `Coordinate` | Board position (x: 0-8, y: 0-9) | `('x', 'y')` |
| `Piece` | Piece with type, side, position | `('piece_type', 'side', 'position')` |
| `Board` | Board state — dict-based piece store | `('pieces',)` |

### Coordinate System (IMPORTANT — counterintuitive)

- **Internal storage**: `x` (column 0-8), `y` (row 0-9) where `y=0` is the **top** of the board.
- **Display (`__repr__`)**: `col(a-i)` + `row(0-9)` where `row=0` is the **bottom** (left-bottom origin, like chess notation).
- **String parsing (`from_string`)**: Accepts `"e0"` format (col a-i, row 0-9, bottom origin), converts internally.

This means `Coordinate(4, 9)` displays as `e0` and `Coordinate(4, 0)` displays as `e9`.

### Move Validation — Dual API

The engine has two parallel paths for move logic:

**Path 1: Direct O(1) check** (`is_move_legal`)
- Used by `Board.move_piece()` for single-move validation.
- Each piece type has a `_check_*_move()` method returning `bool`.
- No list allocation, no enumeration.

**Path 2: Generator enumeration** (`get_possible_moves`)
- Returns a generator yielding `Coordinate` objects.
- Each piece type has a `_generate_*_moves()` method using piece-specific logic (not brute-force).
- Supports early exit via `break`.

Each piece type implements both `_check_*_move()` and `_generate_*_moves()` separately — they share direction constants but have different control flow.

### Board State Management

- `Board.pieces` is a `dict[Coordinate, Piece]` — the single source of truth.
- `get_piece_at(pos)` → `Optional[Piece]` — dict lookup.
- `is_empty(pos)` → `bool` — checks `pos not in pieces`.
- `can_move_to(pos, side)` → `bool` — checks occupancy + ownership (single dict lookup).
- `move_piece(from, to)` → `Optional[Piece]` — validates via `is_move_legal`, removes from old pos, updates piece position, inserts at new pos. Returns captured piece.

### Board Display

- `Board.__repr__()` renders a grid with `+---+` borders.
- Chinese characters are used for piece display (帅/将, 士/仕, 象/相, 马, 车, 炮, 兵/卒) with 红/黑 prefix.
- `_get_display_width()` accounts for CJK characters being 2 columns wide.
- `_pad_cell()` uses display width (not string length) for alignment.

---

## Testing Approach

**No test framework** — tests are bare Python scripts with `assert` statements.

```
python test/chess_test.py          # All 12 tests in one file
```

Key patterns:
- Tests print progress with `=== Test N: Title ===` separators.
- Tests use `assert` (not `unittest`/`pytest`) — any exception fails the test.
- `run_all_tests()` iterates over a list of (name, function) tuples, catching failures.
- Test imports use `from src.chess import ...` (no sys.path hack needed since the project root is accessible).

---

## Conventions & Patterns

### Code Style
- **All variables typed**: Every variable has full type annotations (e.g., `dx: int`, `board_str: str`).
- **`from __future__ import annotations`**: Enables forward references without string quotes.
- **Modern syntax**: `list[...]` not `List[...]`, `dict[...]` not `Dict[...]`.
- **`Optional` retained**: Used for optional return types (e.g., `Optional[Piece]`).
- **`__slots__` on all classes**: Prevents `__dict__`, reduces memory.

### Import Pattern
```python
# From project root
from src.chess import Board, Coordinate, Piece, PieceType, Side

# From test/ directory
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.chess import Board, Coordinate
```

### Movement Constants
All direction tuples are module-level, private, and immutable (tuples, not lists):
```
_KING_DIRS, _ADVISOR_DIRS, _ELEPHANT_DIRS, _HORSE_MOVES, _ORTHOGONAL_DIRS
```

### Naming
- **Methods**: `snake_case` with `_` prefix for private (`_check_king_move`, `_generate_king_moves`).
- **Classes**: `PascalCase`.
- **Enums**: `UPPER_CASE`.
- **Module-level constants**: `_UPPER_CASE` with underscore prefix.

---

## Key Gotchas

1. **Import root requirement**: All imports assume `chess/` is the working directory. Running from `src/` or `test/` will break imports.
2. **Coordinate y-axis is inverted**: Internal `y=0` = top of board, but display `row=0` = bottom. Don't confuse coordinate display format with internal storage.
3. **No test framework**: Don't try to run `pytest test/` — it won't work. Tests are standalone scripts.
4. **`move_piece` uses `is_move_legal` not `get_possible_moves`**: If you add a new piece type, you must implement BOTH `_check_*_move()` and `_generate_*_moves()` — one is used for validation, the other for enumeration.
5. **Elephant eye blocking**: Elephant moves check the midpoint ("eye") — if occupied, the move is blocked. This is NOT the same as horse leg blocking.
6. **Cannon capture is unique**: Cannon moves like a chariot but captures by jumping over exactly one piece. The `_check_cannon_move` and `_generate_cannon_moves` implementations differ significantly from chariot.
7. **`__slots__` prevents dynamic attributes**: You cannot set `piece.something = value` without declaring it in `__slots__`.
8. **No CI/CD, no linting config**: There's no `.pylintrc`, `.mypy.ini`, or CI pipeline.

---

## Doc Organization

- `docs/PROJECT_STRUCTURE.md` — File layout, run commands (Chinese/English).
- `docs/README.md` — Full Chinese-language documentation with class details.
- `docs/OPTIMIZATION.md` — Memory optimizations (`__slots__`, constants, display caching).
- `docs/LAZY_EVALUATION.md` — Generator-based lazy evaluation, dual-API design, performance comparisons.

---

## FAQ for Agents

**Q: How do I add a new test?**
A: Add a `test_*` function to `chess_test.py`, then add it to the `tests` list in `run_all_tests()`.

**Q: How do I add a new piece type?**
A: Add to `PieceType` enum, then implement `_check_*_move()` and `_generate_*_moves()` in `Piece`, update `is_move_legal()` and `get_possible_moves()` dispatch, add display name in `Board._PIECE_DISPLAY_NAMES`, and add initial positions to `Board._init_default_setup()` if needed.

**Q: How do I add a new feature/fix?**
A: Edit `src/chess.py` only — all core logic is there. Tests go in `test/chess_test.py`. Run `python test/chess_test.py` to verify.
