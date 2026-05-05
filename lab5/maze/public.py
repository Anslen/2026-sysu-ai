from __future__ import annotations
from enum import Enum
from typing import Optional, Iterator, cast
from dataclasses import dataclass

class Direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    @staticmethod
    def __iter__() -> Iterator[Direction]:
        return iter([Direction.LEFT, Direction.UP, Direction.RIGHT, Direction.DOWN])

class Position:
    _row: int
    _column: int
    _maze: Maze

    def __init__(self, data: tuple[int, int], maze: Maze) -> None:
        self._row = data[0]
        self._column = data[1]
        self._maze = maze

    def __str__(self) -> str:
        return f"({self._row}, {self._column})"
    
    __repr__ = __str__

    def row(self) -> int:
        return self._row
    
    def column(self) -> int:
        return self._column

    def move(self, direction: Direction) -> Optional[Position]:
        new_data: list[int] = [self._row, self._column]
        if direction == Direction.LEFT:
            new_data[1] -= 1
        elif direction == Direction.UP:
            new_data[0] -= 1
        elif direction == Direction.RIGHT:
            new_data[1] += 1
        elif direction == Direction.DOWN:
            new_data[0] += 1
        
        if (new_data[0] < 0) or (new_data[0] >= self._maze.rows()):
            return None
        if (new_data[1] < 0) or (new_data[1] >= self._maze.columns()):
            return None
        return Position((new_data[0], new_data[1]), self._maze)

class Path:
    _data: list[list[bool]]
    _length: int

    def __str__(self) -> str:
        ret: str = ""
        for each in self._data:
            for char in each:
                ret += "1" if char else "0"
            ret += "\n"
        return ret

    __repr__ = __str__

    def __init__(self, rows: int, columns: int) -> None:
        self._data = [[False] * columns for _ in range(rows)]
        self._length = 0

    def __setitem__(self, key: Position, value: bool) -> None:
        if self[key]:
            if value == False:
                self._length -= 1
        else:
            if value == True:
                self._length += 1
        self._data[key.row()][key.column()] = value

    def __getitem__(self, key: tuple[int, int]|Position) -> bool:
        if isinstance(key, tuple):
            tuple_key: tuple[int, int] = cast(tuple[int, int], key)
            return self._data[tuple_key[0]][tuple_key[1]]

        posi_key: Position = cast(Position, key)
        return self._data[posi_key.row()][posi_key.column()]

    def length(self) -> int:
        return self._length

@dataclass
class State:
    position: Position
    direction: Direction
    path: Path
    
    def __iter__(self) -> Iterator:
        return iter((self.position, self.direction, self.path))

class Maze:
    _data: tuple[str, ...]

    def __init__(self, data: tuple[str, ...]) -> None:
        self._data = data

    def __str__(self) -> str:
        ret: str = ""
        for row in self._data:
            for char in row:
                if char == "1":
                    ret += f"\033[34m{char}\033[0m"
                elif char == "0":
                    ret += f"\033[32m{char}\033[0m"
                else:
                    ret += f"\033[31m{char}\033[0m"
            ret += "\n"
        return ret

    __repr__ = __str__

    def rows(self) -> int:
        return len(self._data)

    def columns(self) -> int:
        return len(self._data[0])

    def start_position(self) -> Position:
        for (i, row) in enumerate(self._data):
            for (j, char) in enumerate(row):
                if char == "S":
                    return Position((i, j), self)
        raise ValueError("No start position found")

    def end_position(self) -> Position:
        for (i, row) in enumerate(self._data):
            for (j, char) in enumerate(row):
                if char == "E":
                    return Position((i, j), self)
        raise ValueError("No end position found")

    def at(self, position: Optional[Position]) -> str:
        if position is None:
            return "1"
        return self._data[position.row()][position.column()]

    def print_path(self, path: Path) -> None:
        for (i, row) in enumerate(self._data):
            for (j, char) in enumerate(row):
                if path[i, j]:
                    print(f"\033[31m{char}\033[0m", end="")
                elif char == "1":
                    print(f"\033[34m{char}\033[0m", end="")
                else:
                    print(f"\033[32m{char}\033[0m", end="")
            print()
