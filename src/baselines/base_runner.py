"""
Abstract protocol that every baseline runner must implement.
Allows ComparisonRunner to treat all systems uniformly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..comparison.metrics import ComparisonSnapshot
from ..knowledge.knowledge_base import KnowledgeBase


class BaselineRunner(ABC):
    """
    Common interface for all conjecture generation systems under comparison.

    Each implementation runs a generate-prove-learn cycle for a fixed
    wall-clock budget and emits ComparisonSnapshots at regular intervals.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in reports (e.g. 'random', 'stp')."""
        ...

    @abstractmethod
    def setup(self, config: dict) -> None:
        """
        Initialise KB, generators, provers from a config dict.
        Called once by ComparisonRunner before run_for().
        """
        ...

    @abstractmethod
    def run_for(
        self,
        wall_clock_budget_seconds: float,
        snapshot_interval_seconds: float = 60.0,
    ) -> List[ComparisonSnapshot]:
        """
        Run for up to wall_clock_budget_seconds, emitting a snapshot every
        snapshot_interval_seconds.  Returns snapshots in chronological order.
        """
        ...

    @abstractmethod
    def get_final_kb(self) -> KnowledgeBase:
        """Return the knowledge base after run_for() has completed."""
        ...
