from public import *
from copy import deepcopy
from typing import cast

class DFS:
    _maze: Maze
    def __init__(self, maze: Maze) -> None:
        self._maze = maze

    def get_path(self) -> Path:
        posi: Position = self._maze.start_position()

        path: Path = Path(self._maze.rows(), self._maze.columns())
        path[posi] = True

        state_stack: list[State] = []

        for direction in Direction:
            if self._maze.at(posi.move(direction)) != "1":
                state_stack.append(State(posi, direction, deepcopy(path)))

        count: int = 1

        while True:
            count += 1
            if len(state_stack) == 0:
                raise ValueError("No path found")

            direction: Direction
            # Pop from stack
            (posi, direction, path) = state_stack.pop()
            posi = cast(Position, posi.move(direction))
            path[posi] = True

            # Check reach end
            if self._maze.at(posi) == "E":
                print(f"DFS finished, {count} nodes visited")
                return path

            # Check next states
            for direction in Direction:
                new_posi: Optional[Position] = posi.move(direction)
                if self._maze.at(new_posi) == "1":
                    continue
                if path[cast(Position, new_posi)]:
                    continue
                state_stack.append(State(posi, direction, deepcopy(path)))

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
    dfs: DFS = DFS(maze)
    path: Path = dfs.get_path()
    maze.print_path(path)
