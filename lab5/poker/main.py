from __future__ import annotations

import threading
import time

from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Self, Iterator, Optional, cast

LOCAL_DICT_BOUND: int = 10

@dataclass
class Poker:
    __slots__ = ("data",)

    data: tuple[list[int], ...]

    @staticmethod
    def read(input: str) -> Poker:
        data: tuple[list[int], ...] = tuple(list(map(int, x.split(","))) for x in input.splitlines())
        return Poker(data)

    def __getitem__(self, key: TakePlace):
        if key.position == TakePlace.Position.BEGIN:
            return self.data[key.row][0]
        else:
            return self.data[key.row][-1]

    def to_tuple(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(row) for row in self.data)
        
    def take(self, take_place: TakePlace, deep_copy: bool) -> Poker:
        """
        Take a card from poker, return new poker state
        
        Defaultly shares data with original poker, set deep_copy to True to get a new copy of data
        """
        new_data: tuple[list[int], ...]
        if deep_copy:
            new_data = tuple(row[:] for row in self.data)
        else:
            new_data = self.data

        if take_place.position == TakePlace.Position.BEGIN:
            new_data[take_place.row].pop(0)
        else:
            new_data[take_place.row].pop()

        return Poker(new_data)

    def put(self, take_place: TakePlace, value: int) -> None:
        """
        Put a card back to poker, modify original poker state
        """
        if take_place.position == TakePlace.Position.BEGIN:
            self.data[take_place.row].insert(0, value)
        else:
            self.data[take_place.row].append(value)

    def is_empty(self) -> bool:
        return sum(map(len, self.data)) == 0

@dataclass
class TakePlace:
    class Position(Enum):
        BEGIN = 0
        END = 1
        def __str__(self) -> str:
            if self == TakePlace.Position.BEGIN:
                return "左端"
            else:
                return "右端"

    __slots__ = ("row", "position")

    row: int
    position: Position

    def __str__(self) -> str:
        return f"第{self.row + 1}行 {self.position}"

    @staticmethod
    def iter(poker: Poker) -> Iterator[TakePlace]:
        for i in range(len(poker.data)):
            if len(poker.data[i]) == 0:
                continue
            yield TakePlace(i, TakePlace.Position.BEGIN)
            if len(poker.data[i]) > 1:
                yield TakePlace(i, TakePlace.Position.END)

class Player(Enum):
    RED = 0
    BLUE = 1

    @staticmethod
    def default() -> Player:
        return Player.RED

    def next(self) -> Player:
        if self == Player.RED:
            return Player.BLUE
        else:
            return Player.RED

class Node:
    CARDS_COUNT: int

    __slots__ = (
        "poker",
        "take_place",
        "taked_value",
        "child",
        "left_cards",
        "player",
        "score",
        "alpha",
        "beta",
        "take_place_iter",
    )

    # Cards state
    poker: Poker
    take_place: Optional[TakePlace]
    taked_value: int

    # Tree state
    child: Optional[Node]
    left_cards: int

    # Player and current red player score
    player: Player
    score: int

    # Alpha-beta pruning
    alpha: float| int
    beta: float| int

    # Iterator of take places, used for spawning children
    take_place_iter: Iterator[TakePlace]

    @staticmethod
    def set_cards_count(count: int) -> None:
        Node.CARDS_COUNT = count

    def __init__(self, poker: Poker) -> None:
        self.poker = poker
        self.take_place = None
        self.taked_value = -1

        self.child = None
        self.left_cards = Node.CARDS_COUNT

        self.player = Player.default()
        self.score = 0

        self.alpha = float('-inf')
        self.beta = float('inf')

        self.take_place_iter = TakePlace.iter(poker)

    @staticmethod
    def from_serial_node(node: SerialNode) -> Node:
        ret: Node = Node(node.poker)
        ret.player = node.player
        ret.score = node.score
        ret.take_place = node.take_place
        ret.taked_value = node.taked_value
        return ret

    def push_to_dict(
            self,
            local_dict: dict[tuple[tuple[int, ...], ...], int],
            global_dict: dict[tuple[tuple[int, ...], ...], int],
            lock: threading.Lock
        ) -> None:
        store_value: int
        if self.player == Player.RED:
            store_value = cast(int, self.alpha) - self.score
        else:
            store_value = cast(int, self.beta) - self.score

        if self.left_cards < LOCAL_DICT_BOUND:
            local_dict[self.poker.to_tuple()] = store_value
            return
        
        with lock:
            global_dict[self.poker.to_tuple()] = store_value

    def get_from_dict(
        self,
        local_dict: dict[tuple[tuple[int, ...], ...], int],
        global_dict: dict[tuple[tuple[int, ...], ...], int],
        lock: threading.Lock
        ) -> Optional[int]:
        if self.left_cards < LOCAL_DICT_BOUND:
            return local_dict.get(self.poker.to_tuple())

        with lock:
            return global_dict.get(self.poker.to_tuple())

    def spawn_child(self, take_place: TakePlace, deep_copy: bool) -> Self:
        taked_value: int = self.poker[take_place]

        # Spawn child, take from place
        child: Self
        if deep_copy:
            child = self.__class__(self.poker.take(take_place, True))
        else:
            child = self.__class__(self.poker.take(take_place, False))
        child.take_place = take_place

        child.score = self.score
        child.taked_value = taked_value
        child.left_cards = self.left_cards - 1
        # Set child state
        if self.player == Player.RED:
            child.player = Player.BLUE
            child.alpha = self.alpha
            child.score += taked_value
        else:
            child.player = Player.RED
            child.beta = self.beta

        return child

    def all_child_nodes(self) -> list[Self]:
        return [self.spawn_child(i, True) for i in self.take_place_iter]

    def put_back_poker(self) -> None:
        """
        Put the card back to poker, modify original poker state
        """
        if self.take_place is not None:
            self.poker.put(self.take_place, self.taked_value)

class SerialNode(Node):
    """
    Tree node for serial search

    This class will store all children and child index
    """

    child_index: int
    children: list[Node]

    def __init__(self, poker: Poker) -> None:
        super().__init__(poker)
        self.child_index = -1
        self.children = []

    def update_alpha_beta(self) -> None:
        """
        Update alpha-beta pruning value by children

        CAUSION: This method only checks child nodes
        """
        if self.player == Player.RED:
            self.child_index, self.alpha = max(
                map(lambda x: (x[0], x[1].beta),enumerate(self.children)),
                key = lambda x: x[1],
                default = (-1, self.score)
            )
        else:
            self.child_index, self.beta = min(
                map(lambda x: (x[0], x[1].alpha),enumerate(self.children)),
                key = lambda x: x[1],
                default = (-1, self.score)
            )

    def put_back_poker(self) -> None:
        # Poker is deep copied, don't need to put back
        pass

class Solution:
    global_solved: dict[tuple[tuple[int, ...], ...], int] = {}
    lock: threading.Lock = threading.Lock()

    @staticmethod
    def solve(poker: Poker, card_count: int) -> str:
        Solution.global_solved.clear()
        root: SerialNode = SerialNode(poker)
        levels: list[list[SerialNode]] = [[root]]

        while len(levels[-1]) < 20:
            next_level: list[SerialNode] = Solution._spawn_next_level(levels[-1])
            if len(next_level) == 0:
                break
            levels.append(next_level)

        """
        # Debug
        for each in levels[-1]:
            Solution._spawn_subtree(each)
        
        """

        # Start multi-threading for last level
        threads: list[threading.Thread] = []
        for node in levels[-1]:
            thread = threading.Thread(target=Solution._spawn_subtree, args=(node,))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        for level in levels[-2::-1]:
            tuple(map(lambda x: x.update_alpha_beta(), level))

        total_score: int = card_count * (card_count + 1) // 2
        take_place: TakePlace = cast(TakePlace, root.children[root.child_index].take_place)

        return f"{take_place} 牌点数{root.poker[take_place]}\n" +\
        f"小红: {root.alpha} 小蓝: {total_score - root.alpha}\n"

    @staticmethod
    def _spawn_next_level(level: list[SerialNode]) -> list[SerialNode]:
        ret: list[SerialNode] = []
        for node in level:
            children: list[SerialNode] = node.all_child_nodes()
            node.children.extend(children)
            ret.extend(children)
        return ret

    @staticmethod
    def _spawn_subtree(node: SerialNode) -> None:
        local_solved: dict[tuple[tuple[int, ...], ...], int] = {}
        node_copy: Node = Node.from_serial_node(node)
        node_stack: list[Node] = [node_copy]

        while len(node_stack) > 0:
            current: Node = node_stack[-1]

            # Reach leaf node
            if current.poker.is_empty():
                # Update score
                current.alpha = current.score
                current.beta = current.score
                current.push_to_dict(local_solved, Solution.global_solved, Solution.lock)

                # Remove from stack
                current.put_back_poker()
                node_stack.pop()
                continue

            # Update alpha-beta pruning by last child
            if current.child is not None:
                if current.player == Player.RED:
                    if current.child.beta > current.alpha:
                        current.alpha = current.child.beta
                else:
                    if current.child.alpha < current.beta:
                        current.beta = current.child.alpha
                current.child = None

            # Alpha-beta pruning
            if current.alpha >= current.beta:
                current.put_back_poker()
                node_stack.pop()
                continue

            # Get next child node
            take_place: TakePlace
            try:
                take_place: TakePlace = next(current.take_place_iter)
            except StopIteration:
                # No more child
                current.push_to_dict(local_solved, Solution.global_solved, Solution.lock)
                current.put_back_poker()
                node_stack.pop()
                continue

            # Spawn child node
            child: Node = current.spawn_child(take_place, False)
            current.child = child

            # Check if child node is already solved
            result: Optional[int]
            result = child.get_from_dict(local_solved, Solution.global_solved, Solution.lock)
            if result is not None:
                if child.player == Player.RED:
                    child.alpha = result + child.score
                else:
                    child.beta = result + child.score
                child.put_back_poker()
                continue

            node_stack.append(child)

        node.alpha = node_copy.alpha
        node.beta = node_copy.beta

@dataclass
class TestFailedException(Exception):
    expected: str
    actual: str

    def __str__(self) -> str:
        return f"Expected:\n{self.expected}\nActual:\n{self.actual}"

def all_test() -> None:
    test_path: Path = Path(__file__).parent / "test_cases"
    passed_count: int = 0
    all_tests: int = len(list(test_path.iterdir()))
    for test_dir in test_path.iterdir():
        if not test_dir.is_dir():
            continue

        try:
            _test_with_path(test_dir)
        except TestFailedException as e:
            print(f"Test case {test_dir.name} failed.\n{e}")
            return

        passed_count += 1
        print(f"\r{passed_count}/{all_tests} passed.")

    print("All test cases passed.")

def single_test(suffix: str) -> None:
    test_path: Path = Path(__file__).parent / "test_cases" / ("test" + suffix)
    try:
        _test_with_path(test_path, output=True)
    except TestFailedException as e:
        print(f"Test case {test_path.name} failed.\n{e}")
        return
    print(f"Test case {suffix} passed.")

def _test_with_path(path: Path, output: bool = False) -> None:
    start_time: float = 0.0
    if output:
        start_time = time.time()

    input_path: Path = path / "input.txt"
    output_path: Path = path / "output.txt"
    with input_path.open() as input_file, output_path.open(encoding="utf-8") as output_file:
        cards_count: int = int(input_file.readline())
        Node.set_cards_count(cards_count)

        input_data: str = input_file.read()
        expected_output: str = output_file.read()
        actual_output: str = Solution.solve(Poker.read(input_data), cards_count)
        if expected_output != actual_output:
            raise TestFailedException(expected_output, actual_output)

    end_time: float = 0.0
    if output:
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    single_test("10")
