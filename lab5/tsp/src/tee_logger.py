"""
TeeLogger — simultaneous console + file output with immediate flush.

Writes every message to both stdout and a timestamped log file under
log/run_<timestamp>/, flushing after each write so partial output
survives a crash or Ctrl‑C.  Also records per‑generation best distances
to a companion CSV file in the same run directory.
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TextIO

LOG_DIR: Path = Path("log")
"""Directory where per‑run subdirectories are stored."""


class TeeLogger:
    """
    Logs messages to both stdout and a persistent log file.

    Each instance creates a uniquely‑named run subdirectory inside
    LOG_DIR containing:
      - tsp.log          : human‑readable text log
      - convergence.csv  : generation,best_distance records

    Every write is flushed immediately.
    """

    __slots__ = ("_log_file", "_csv_file", "_log_dir", "_start_time")

    _log_file: TextIO
    _csv_file: TextIO
    _log_dir: Path
    _start_time: float

    def __init__(self, log_dir: Path | None = None) -> None:
        """
        Create a run subdirectory and open the log + CSV files for writing.

        Args:
            log_dir: Optional override for the parent log directory.
                     Defaults to LOG_DIR ("log/").
        """
        parent_dir: Path = log_dir if log_dir is not None else LOG_DIR
        timestamp: str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._log_dir = parent_dir / f"run_{timestamp}"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        log_path: Path = self._log_dir / "tsp.log"
        self._log_file = open(str(log_path), "w", encoding="utf-8")

        csv_path: Path = self._log_dir / "convergence.csv"
        self._csv_file = open(str(csv_path), "w", encoding="utf-8", newline="")
        self._csv_file.write("generation,best_distance\n")
        self._csv_file.flush()

        self._start_time = time.time()

    # -- properties ------------------------------------------------------------

    @property
    def log_dir(self) -> Path:
        """Path to the run subdirectory containing all outputs."""
        return self._log_dir

    @property
    def elapsed(self) -> float:
        """Return seconds since this logger was created (wall‑clock)."""
        return time.time() - self._start_time

    # -- public API -----------------------------------------------------------

    def log(self, message: str) -> None:
        """Write *message* to stdout and the log file, then flush both."""
        sys.stdout.write(message + "\n")
        sys.stdout.flush()
        self._log_file.write(message + "\n")
        self._log_file.flush()

    def record_generation(self, generation: int, best_distance: float) -> None:
        """
        Append a generation record to the CSV file and flush immediately.

        Args:
            generation: The current generation number (zero‑based).
            best_distance: The best tour distance found so far.
        """
        self._csv_file.write(f"{generation},{best_distance:.6f}\n")
        self._csv_file.flush()

    # -- header / footer helpers ----------------------------------------------

    def log_header(
        self,
        *,
        command: str = "",
        input_file: str = "",
        cities: int = 0,
        pop_size: int = 0,
        max_gen: int = 0,
        mutation_methods: dict[str, float] | None = None,
        crossover_method: str = "",
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
            f"Run directory: {self._log_dir}",
            f"Command: {command}" if command else "",
            f"Input file: {input_file}",
            f"Cities: {cities}",
            "",
            "Algorithm Parameters:",
            f"  Population Size  : {pop_size}",
            f"  Max Generations  : {max_gen}",
            f"  Crossover Method : {crossover_method}",
            "  Mutation Methods :",
        ]
        if mutation_methods:
            for method, rate in mutation_methods.items():
                lines.append(f"    {method:15s} : {rate}")
        else:
            lines.append("    (none)")
        lines.extend([
            f"  Tournament Size  : {tournament_size}",
            f"  Elitism Count    : {elitism_count}",
            f"  Report Interval  : {report_interval}",
            f"  Random Seed      : {seed}",
            f"  Thread Workers   : {thread_workers}",
            sep,
            "",
        ])
        for line in lines:
            if line:  # skip empty strings so they don't add blank lines
                self.log(line)

    def log_footer(self, *, best_tour: str = "", best_distance: float = 0.0) -> None:
        """
        Write a footer with total runtime and final results.
        """
        sep: str = "=" * 60
        elapsed_sec: float = self.elapsed
        lines: list[str] = [
            sep,
            f"Finished at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Total elapsed: {elapsed_sec:.1f}s",
            f"Best distance: {best_distance:.2f}",
            f"Best tour ({len(best_tour.split(',')) if best_tour else 0} cities):",
            f"  {best_tour}",
            sep,
        ]
        for line in lines:
            self.log(line)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Flush and close both the log file and CSV file.  Safe to call multiple times."""
        if not self._log_file.closed:
            self._log_file.close()
        if not self._csv_file.closed:
            self._csv_file.close()

    def __enter__(self) -> TeeLogger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
