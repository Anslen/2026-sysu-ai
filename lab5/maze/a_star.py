from __future__ import annotations

import heapq

from public import Position, Direction, Maze, Path
from dataclasses import dataclass
from copy import deepcopy
from typing import cast, Iterator, Optional

@dataclass
class State:
    posi: Position
    direction: Direction
    path: Path
    heuristic: int

    def __iter__(self) -> Iterator:
        return iter((self.posi, self.direction, self.path))

    def __lt__(self, other: State) -> bool:
        return self.heuristic < other.heuristic

class AStar:
    _maze: Maze
    _start_position: Position
    _end_position: Position

    def __init__(self, maze: Maze) -> None:
        self._maze = maze
        self._start_position = maze.start_position()
        self._end_position = maze.end_position()

    def _heuristic(self, posi: Position, path: Path) -> int:
        return abs(self._end_position.row() - posi.row()) + abs(self._end_position.column() - posi.column()) + path.length()

    def get_path(self) -> Path:
        posi: Position = self._start_position

        path: Path = Path(self._maze.rows(), self._maze.columns())
        path[posi] = True

        state_heap: list[State] = []

        for direction in Direction:
            new_posi: Optional[Position] = posi.move(direction)
            if self._maze.at(new_posi) != "1":
                state: State = State(posi, direction, deepcopy(path), self._heuristic(cast(Position, new_posi), path))
                heapq.heappush(state_heap, state)

        count: int = 1

        while True:
            count += 1
            if len(state_heap) == 0:
                raise ValueError("No path found")

            direction: Direction
            # Pop from heap
            (posi, direction, path) = heapq.heappop(state_heap)
            posi = cast(Position, posi.move(direction))
            path[posi] = True

            # Check reach end
            if self._maze.at(posi) == "E":
                print(f"A* finished, {count} nodes visited")
                return path

            # Check next states
            for direction in Direction:
                new_posi: Optional[Position] = posi.move(direction)
                if self._maze.at(new_posi) == "1":
                    continue
                if path[cast(Position, new_posi)]:
                    continue
                state: State = State(posi, direction, deepcopy(path), self._heuristic(cast(Position, new_posi), path))
                heapq.heappush(state_heap, state)

if __name__ == "__main__":
    maze: Maze = Maze((
        "S00000001000000",
        "111111010111110",
        "000000010100000",
        "011111110101111",
        "010000000001000",
        "010111111111010",
        "010000000001010",
        "000101111101010",
        "111101000101010",
        "000001010101010",
        "011111010101010",
        "010000010100010",
        "011111110111100",
        "000000000100001",
        "11111111110000E"
    ))
    a_star: AStar = AStar(maze)
    path: Path = a_star.get_path()
    maze.print_path(path)
