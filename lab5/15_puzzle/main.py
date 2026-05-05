from __future__ import annotations

import heapq
import random
import threading

from copy import deepcopy
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable, Self

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    INVALID = 4

    def __str__(self) -> str:
        if self == Direction.UP:
            return "UP"
        elif self == Direction.RIGHT:
            return "RIGHT"
        elif self == Direction.DOWN:
            return "DOWN"
        elif self == Direction.LEFT:
            return "LEFT"
        else:
            return "INVALID"

    __repr__ = __str__

    @staticmethod
    def directions() -> Iterator[Direction]:
        return iter((Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT))

    def is_opposite(self, other: Direction) -> bool:
        if self == Direction.UP:
            return other == Direction.DOWN
        elif self == Direction.RIGHT:
            return other == Direction.LEFT
        elif self == Direction.DOWN:
            return other == Direction.UP
        elif self == Direction.LEFT:
            return other == Direction.RIGHT
        else:
            return False

@dataclass
class Position:
    __slots__ = ("index",)
    index: int

    def move(self, direction: Direction) -> Optional[Position]:
        match direction:
            case Direction.UP:
                if self.index < 4:
                    return None
                return Position(self.index - 4)

            case Direction.RIGHT:
                new_index = self.index + 1
                if new_index % 4 == 0:
                    return None
                return Position(new_index)

            case Direction.DOWN:
                if self.index >= 12:
                    return None
                return Position(self.index + 4)

            case Direction.LEFT:
                if self.index % 4 == 0:
                    return None
                return Position(self.index - 1)

            case _:
                raise ValueError("Invalid direction")

@dataclass
class Puzzle:
    __slots__ = ("_data", "_hole_position", "_heuristic_func", "_heuristic", "_visited")

    _data: tuple[int, ...]
    _hole_position: Position
    _heuristic_func: Callable[[Puzzle, Position], int]
    _heuristic: int
    _visited: set[tuple[int, ...]]

    @staticmethod
    def random(heuristic_func: Callable[[Puzzle, Position], int], shuffle_count: int) -> Puzzle:
        ret: Puzzle = Puzzle(tuple(range(16)), heuristic_func)
        for _ in range(shuffle_count):
            direction: Direction = random.choice(list(Direction.directions()))
            new_puzzle: Optional[Puzzle] = ret.move(direction, True)
            if new_puzzle is not None:
                ret = new_puzzle
        return Puzzle(ret._data, heuristic_func)

    def __init__(self, data: tuple[int, ...], heuristic_func: Callable[[Puzzle, Position], int], _inner_created: bool = False):
        self._data = data
        self._heuristic_func = heuristic_func

        self._heuristic = 0
        if _inner_created:
            return

        self._visited = {data}
        # Calculate heuristic and hole position
        self._hole_position = Position(data.index(15))
        for i in range(16):
            self._heuristic += heuristic_func(self, Position(i))

    def __str__(self) -> str:
        ret: str = ""
        for row in range(4):
            for col in range(4):
                value: int = self._data[row * 4 + col]
                ret += f"{value:>3d} "
            ret += "\n"
        return ret

    def __getitem__(self, key: Position):
        return self._data[key.index]

    def __lt__(self, other: Puzzle) -> bool:
        return self._heuristic < other._heuristic

    def move(self, direction: Direction, _shuffle_mode: bool = False) -> Optional[Puzzle]:
        def move_generator(
            old_data: tuple[int, ...],
            old_hole_position: Position,
            new_hole_position: Position
            ) -> Callable[[int], int]:

            def new_value(index: int) -> int:
                if index == old_hole_position.index:
                    return old_data[new_hole_position.index]
                elif index == new_hole_position.index:
                    return 15
                else:
                    return old_data[index]
            return new_value

        # Get new hole position
        optional_new_hole_position: Optional[Position] = self._hole_position.move(direction)
        if optional_new_hole_position is None:
            return None

        # Move hole 
        new_hole_position: Position = optional_new_hole_position
        new_data: tuple[int, ...] = tuple(move_generator(
            self._data,
            self._hole_position,
            new_hole_position)(i) for i in range(16))

        # Check if visited
        if (not _shuffle_mode) and (new_data in self._visited):
            return None

        self._visited.add(new_data)
        ret: Puzzle = Puzzle(new_data, self._heuristic_func, True)

        # Set visited and hole position
        ret._visited = self._visited
        ret._hole_position = new_hole_position

        # Update heuristic
        ret._heuristic = self._heuristic
        ret._heuristic -= self._heuristic_func(self, new_hole_position)
        ret._heuristic += self._heuristic_func(ret, self._hole_position)
        # Solution length
        if not _shuffle_mode:
            ret._heuristic += 1
        return ret
    
    def is_solved(self) -> bool:
        for i in range(15):
            if self._data[i] != i:
                return False
        return True

class AStar:
    @dataclass
    class State:
        puzzle: Puzzle
        parent: Optional[AStar.State]
        last_direction: Direction = Direction.INVALID

        def __lt__(self, other: Self) -> bool:
            return self.puzzle < other.puzzle

    @staticmethod
    def get_solution(puzzle: Puzzle, print_info: bool = False) -> tuple[list[Direction], int]:
        initial_state = AStar.State(puzzle, None)
        state_heap: list[AStar.State] = [initial_state]

        count: int = 1
        while True:
            current_state: AStar.State = heapq.heappop(state_heap)
            current_puzzle: Puzzle = current_state.puzzle
            last_direction: Direction = current_state.last_direction

            count += 1
            if print_info and count % 10000 == 0:
                print(f"Checked {count} states, current heuristic: {current_puzzle._heuristic}")

            if current_puzzle.is_solved():
                # Backtrack to build the solution path
                solution: list[Direction] = []
                node = current_state
                while node is not None and node.parent is not None:
                    solution.append(node.last_direction)
                    node = node.parent
                solution.reverse()
                #print(f"Solved in {len(solution)} moves, checked {count} states")
                return solution, count

            for direction in Direction.directions():
                if direction.is_opposite(last_direction):
                    continue
                new_puzzle: Optional[Puzzle] = current_puzzle.move(direction)
                if new_puzzle is None:
                    continue

                heapq.heappush(state_heap, AStar.State(new_puzzle, current_state, direction))

def manhattan_distance(puzzle: Puzzle, position: Position) -> int:
    value: int = puzzle[position]
    if value == 15:
        return 0
    
    target_row: int = value // 4
    target_col: int = value % 4
    current_row: int = position.index // 4
    current_col: int = position.index % 4
    return abs(current_row - target_row) + abs(current_col - target_col)

def astar_test(test_time: int, shuffle_count: int):
    costs: list[int] = []
    finished_count: int = 0
    lock: threading.Lock = threading.Lock()
    
    def run_single_test():
        puzzle: Puzzle = Puzzle.random(manhattan_distance, shuffle_count)
        _, cost = AStar.get_solution(puzzle)
        with lock:
            nonlocal finished_count
            costs.append(cost)
            finished_count += 1
            print(f"\r{finished_count}/{test_time} tests completed", end="", flush=True)
    
    threads: list[threading.Thread] = []
    for _ in range(test_time):
        thread: threading.Thread = threading.Thread(target=run_single_test)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

    cost_sum: int = sum(costs)
    print(f"\n{test_time} tests completed, shuffle count: {shuffle_count}, average cost: {cost_sum / test_time: .2f}")

def random_and_solve_test(shuffle_count: int):
    puzzle: Puzzle = Puzzle.random(manhattan_distance, shuffle_count)
    print("Initial state:")
    print(puzzle)
    solution, cost = AStar.get_solution(puzzle, print_info=True)
    print(f"Solved in {len(solution)} moves, cost: {cost}")
    print("Solution path:")
    for move in solution:
        print(move)

if __name__ == "__main__":
    astar_test(100, 200)
