"""Compact object-message data structures and deterministic fixtures."""

from .adequacy import AdequacyReport, AdequacyStatus, evaluate_adequacy
from .download import DownloadDecision, fetch_next_data
from .manifest import Catalog, CatalogEntry, DataRole, ManifestRecord
from .objects import ObjectMessage, ObjectSource, PerceivedObject
from .smoke import SmokeFixture, SmokeFrame, generate_smoke_fixture
from .splits import Partition, SplitAssignment, assign_intersections

__all__ = [
    "AdequacyReport",
    "AdequacyStatus",
    "Catalog",
    "CatalogEntry",
    "DataRole",
    "DownloadDecision",
    "ManifestRecord",
    "ObjectMessage",
    "ObjectSource",
    "Partition",
    "PerceivedObject",
    "SmokeFixture",
    "SmokeFrame",
    "SplitAssignment",
    "assign_intersections",
    "evaluate_adequacy",
    "fetch_next_data",
    "generate_smoke_fixture",
]
