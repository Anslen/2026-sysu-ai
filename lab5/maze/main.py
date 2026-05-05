from public import Maze, Path
from dfs import DFS
from bfs import BFS
from a_star import AStar

def main() -> None:
    # Read from maze.txt
    maze_data: list[str]
    with open("maze.txt", "r") as f:
        maze_data = f.read().splitlines()
    maze: Maze = Maze(tuple(maze_data))

    print("Maze:")
    print(maze)

    dfs: DFS = DFS(maze)
    dfs_path: Path = dfs.get_path()
    print("DFS path:")
    maze.print_path(dfs_path)
    print(f"Path length: {dfs_path.length()}")
    print()

    bfs: BFS = BFS(maze)
    bfs_path: Path = bfs.get_path()
    print("BFS path:")
    maze.print_path(bfs_path)
    print(f"Path length: {bfs_path.length()}")
    print()

    a_star: AStar = AStar(maze)
    a_star_path: Path = a_star.get_path()
    print("A* path:")
    maze.print_path(a_star_path)
    print(f"Path length: {a_star_path.length()}")
    print()

if __name__ == "__main__":
    main()
