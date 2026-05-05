from typing import cast, Iterator, Optional
import threading

from src.chess import Board, Coordinate, Side
from src.state import GameState, MoveRecord

# Global constant: nodes with left_depth > this threshold store results
# in the thread-local cache to avoid lock contention on the global cache.
LOCAL_CACHE_MIN_DEPTH: int = 10

# Type alias for cache entries: (alpha, beta, left_depth)
_CacheEntry = tuple[int, int, int]

# Global cache shared across all threads
_global_cache: dict[tuple, _CacheEntry] = {}
_global_cache_lock: threading.Lock = threading.Lock()

# Thread-local storage for per-thread caches
_thread_local: threading.local = threading.local()


def _get_thread_local_cache() -> dict[tuple, _CacheEntry]:
    """Get or create the thread-local cache for the current thread."""
    if not hasattr(_thread_local, 'cache'):
        _thread_local.cache = {}
    return _thread_local.cache


def _make_cache_key(board: Board, current_turn: Side) -> tuple:
    """Create a hashable cache key from board state and current turn."""
    pieces_key = frozenset(
        (p.piece_type.value, p.side.value, p.position.x, p.position.y)
        for p in board._pieces.values()
    )
    return (pieces_key, current_turn.value)


def _cache_lookup(game_state: GameState) -> Optional[_CacheEntry]:
    """Look up a cached result for the given game state.

    Returns (alpha, beta, left_depth) if a cached result with sufficient
    search depth exists, None otherwise.
    """
    key = _make_cache_key(game_state.board, game_state.current_turn)

    # Check thread-local cache first
    local_cache = _get_thread_local_cache()
    if key in local_cache:
        cached_alpha, cached_beta, cached_depth = local_cache[key]
        if cached_depth >= game_state.left_depth:
            return (cached_alpha, cached_beta, cached_depth)

    # Check global cache
    with _global_cache_lock:
        if key in _global_cache:
            cached_alpha, cached_beta, cached_depth = _global_cache[key]
            if cached_depth >= game_state.left_depth:
                return (cached_alpha, cached_beta, cached_depth)

    return None


def _cache_store(game_state: GameState) -> None:
    """Store the current game state's minimax result in the appropriate cache.

    Results with left_depth > LOCAL_CACHE_MIN_DEPTH go to the thread-local
    cache; shallower results go to the global cache.
    """
    key = _make_cache_key(game_state.board, game_state.current_turn)
    entry = (game_state.alpha, game_state.beta, game_state.left_depth)

    if game_state.left_depth > LOCAL_CACHE_MIN_DEPTH:
        # Store in thread-local cache to avoid lock contention
        local_cache = _get_thread_local_cache()
        # Only update if the new result comes from a deeper search
        if key in local_cache:
            _, _, existing_depth = local_cache[key]
            if existing_depth >= game_state.left_depth:
                return
        local_cache[key] = entry
    else:
        # Store in global cache
        with _global_cache_lock:
            if key in _global_cache:
                _, _, existing_depth = _global_cache[key]
                if existing_depth >= game_state.left_depth:
                    return
            _global_cache[key] = entry


def solve(board: Board, ai_side: Side) -> tuple[Coordinate, Coordinate]:
    """
    Solve the given board using minimax search and return the best move (from, to).
    GameState is constructed internally so the entire search tree is released when solve() returns.
    """
    # Clear global cache from previous solve() calls to prevent unbounded memory growth
    _global_cache.clear()

    game_state: GameState = GameState.from_board(board, ai_side)

    children: list[GameState] = []
    for piece in game_state.board.get_pieces_by_side(game_state.current_turn):
        for to_pos in piece.get_possible_moves(game_state.board):
            new_state: GameState = game_state.apply_move(piece, to_pos, deep_copy=True)
            children.append(new_state)

    # Use multithreading to parallelize spawn_subtree calls
    threads: list[threading.Thread] = []
    for each in children:
        thread = threading.Thread(target=spawn_subtree, args=(each,))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Children are always the opponent's turn.
    # Select child prioritizing king capture, then king safety, then score:
    #   For RED (max): prefer black_king_captured, then red_king safe, then higher beta.
    #   For BLACK (min): prefer red_king_captured, then black_king safe, then lower alpha.
    best_child: GameState
    if game_state.current_turn == Side.RED:
        best_child = max(children, key=lambda c: (c.black_king_captured, not c.red_king_captured, c.beta, c.result_left_depth))
    else:
        best_child = min(children, key=lambda c: (not c.red_king_captured, c.black_king_captured, c.alpha, -c.result_left_depth))

    best_move: MoveRecord = cast(MoveRecord, best_child.last_move)
    return best_move.from_pos, best_move.to_pos

def spawn_subtree(game_state: GameState):
    """
    Solve the given game state using minimax search 
    
    Result will be writed to game_state
    """
    state_stack: list[GameState] = [game_state]

    while len(state_stack) > 0:
        current: GameState = state_stack[-1]

        # Check cache on first encounter of this node
        if current.child is None and current.checking_piece is None:
            # If own king already captured, this is a losing position
            if current.red_king_captured or current.black_king_captured:
                current.result_left_depth = current.left_depth
                _cache_store(current)
                current.recover()
                state_stack.pop()
                continue

            cached = _cache_lookup(current)
            if cached is not None:
                current.alpha, current.beta, current.result_left_depth = cached
                current.recover()
                state_stack.pop()
                continue

        # Update alpha-beta by child's value
        if current.child is not None:
            if current.current_turn == Side.RED:
                # Maximizing player: prefer king-safe children first
                is_first = (current.result_left_depth == -1)
                child_better = is_first
                if not is_first:
                    child_captured = current.child.red_king_captured
                    best_captured = current.red_king_captured
                    if child_captured and not best_captured:
                        child_better = False
                    elif not child_captured and best_captured:
                        child_better = True
                    elif current.child.beta > current.alpha:
                        child_better = True
                    elif current.child.beta == current.alpha and current.child.result_left_depth > current.result_left_depth:
                        child_better = True

                if child_better:
                    current.alpha = current.child.beta
                    current.red_king_captured = current.child.red_king_captured
                    current.result_left_depth = current.child.result_left_depth

            else:
                # Minimizing player: prefer king-safe children first
                is_first = (current.result_left_depth == -1)
                child_better = is_first
                if not is_first:
                    child_captured = current.child.black_king_captured
                    best_captured = current.black_king_captured
                    if child_captured and not best_captured:
                        child_better = False
                    elif not child_captured and best_captured:
                        child_better = True
                    elif current.child.alpha < current.beta:
                        child_better = True
                    elif current.child.alpha == current.beta and current.child.result_left_depth > current.result_left_depth:
                        child_better = True

                if child_better:
                    current.beta = current.child.alpha
                    current.black_king_captured = current.child.black_king_captured
                    current.result_left_depth = current.child.result_left_depth

            # Check for alpha-beta cutoff or king captured
            if current.alpha >= current.beta or current.red_king_captured or current.black_king_captured:
                current.recover()
                state_stack.pop()
                continue

        if current.left_depth == 0:
            current.alpha = current.score
            current.beta = current.score
            current.result_left_depth = current.left_depth
            _cache_store(current)
            current.recover()
            state_stack.pop()
            continue

        try:
            if current.checking_piece is None:
                # Initialize move generation for the current piece
                current.checking_piece = next(current.piece_iter)
                current.move_iter = current.checking_piece.get_possible_moves(current.board)

            try:
                # Get the next move for the current piece
                to_pos: Coordinate = next(cast(Iterator[Coordinate], current.move_iter))

                # Create a new game state for this move
                new_state: GameState = current.apply_move(current.checking_piece, to_pos)
                current.child = new_state
                state_stack.append(new_state)

                current.has_vaild_move = True

            except StopIteration:
                # No more moves for the current piece, reset and move to the next piece
                current.checking_piece = next(current.piece_iter)
                current.move_iter = current.checking_piece.get_possible_moves(current.board)

        except StopIteration:
            # No more pieces to check, backtrack
            if not current.has_vaild_move:
                # If no valid moves were found, this is a losing position
                if current.current_turn == Side.RED:
                    current.red_king_captured = True
                else:
                    current.black_king_captured = True
                current.result_left_depth = current.left_depth
            _cache_store(current)
            current.recover()
            state_stack.pop()
