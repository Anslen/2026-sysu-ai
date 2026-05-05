"""
TeeLogger — simultaneous console + file output with immediate flush.

Writes every message to both stdout and a timestamped log file under log/,
flushing after each write so partial output survives a crash or Ctrl‑C.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TextIO


LOG_DIR: Path = Path("log")
"""Directory where log files are stored."""


class TeeLogger:
    """
    Logs messages to both stdout and a persistent log file.

    Each instance opens a uniquely‑named log file (timestamped) inside
    LOG_DIR and writes every message to both sinks, flushing immediately.

    Usage as context manager::

        with TeeLogger() as log:
            log.log("Processing ...")

    Usage without context manager::

        log = TeeLogger()
        log.log("Processing ...")
        log.close()
    """

    __slots__ = ("_file", "_filepath", "_start_time")

    _file: TextIO
    _filepath: Path
    _start_time: float

    def __init__(self, log_dir: Path | None = None) -> None:
        """
        Create a log file and prepare for dual output.

        Args:
            log_dir: Optional override for the log directory.
                     Defaults to LOG_DIR ("log/").
        """
        target_dir: Path = log_dir if log_dir is not None else LOG_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._filepath = target_dir / f"tsp_{timestamp}.log"
        self._file = open(str(self._filepath), "w", encoding="utf-8")
        self._start_time = time.time()

    # -- public API -----------------------------------------------------------

    def log(self, message: str) -> None:
        """Write *message* to stdout and the log file, then flush both."""
        sys.stdout.write(message + "\n")
        sys.stdout.flush()
        self._file.write(message + "\n")
        self._file.flush()

    def elapsed(self) -> float:
        """Return seconds since this logger was created (wall‑clock)."""
        return time.time() - self._start_time

    @property
    def filepath(self) -> Path:
        """Path to the currently open log file."""
        return self._filepath

    @property
    def start_time(self) -> float:
        """time.time() value captured at logger creation."""
        return self._start_time

    # -- header / footer helpers ----------------------------------------------

    def log_header(
        self,
        *,
        command: str = "",
        input_file: str = "",
        cities: int = 0,
        pop_size: int = 0,
        max_gen: int = 0,
        mutation_rate: float = 0.0,
        crossover_method: str = "",
        mutation_method: str = "",
        tournament_size: int = 0,
        elitism_count: int = 0,
        report_interval: int = 0,
        seed: int | None = None,
        thread_workers: int = 0,
    ) -> None:
        """
        Write a human‑readable header with run metadata and GA parameters.

        All parameters are keyword‑only to keep call sites self‑documenting.
        """
        sep: str = "=" * 60
        lines: list[str] = [
            sep,
            "TSP Genetic Algorithm Solver",
            f"Start time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Command: {command}" if command else "",
            f"Input file: {input_file}",
            f"Cities: {cities}",
            "",
            "Algorithm Parameters:",
            f"  Population Size  : {pop_size}",
            f"  Max Generations  : {max_gen}",
            f"  Mutation Rate    : {mutation_rate}",
            f"  Crossover Method : {crossover_method}",
            f"  Mutation Method  : {mutation_method}",
            f"  Tournament Size  : {tournament_size}",
            f"  Elitism Count    : {elitism_count}",
            f"  Report Interval  : {report_interval}",
            f"  Random Seed      : {seed}",
            f"  Thread Workers   : {thread_workers}",
            sep,
            "",
        ]
        for line in lines:
            if line:  # skip empty strings so they don't add blank lines
                self.log(line)

    def log_footer(self, *, best_tour: str = "", best_distance: float = 0.0) -> None:
        """
        Write a footer with total runtime and final results.
        """
        sep: str = "=" * 60
        elapsed: float = self.elapsed()
        lines: list[str] = [
            sep,
            f"Finished at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Total elapsed: {elapsed:.1f}s",
            f"Best distance: {best_distance:.2f}",
            f"Best tour ({len(best_tour.split(',')) if best_tour else 0} cities):",
            f"  {best_tour}",
            sep,
        ]
        for line in lines:
            self.log(line)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the log file. Safe to call multiple times."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> TeeLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
