"""River path grouping utilities."""

from typing import Dict, List

from src.types import RiverPoint


def group_rivers_by_sequence(
    river_points: List[RiverPoint],
) -> Dict[int, List[RiverPoint]]:
    """
    Group RiverPoint objects into ordered semantic river paths per sequence.

    Returns:
        dict: sequence_id -> list of RiverPoint sorted by position.
    """
    rivers: Dict[int, List[RiverPoint]] = {}

    for rp in river_points:
        rivers.setdefault(rp.sequence_id, []).append(rp)

    # Sort each river by position (ascending)
    for seq_id, points in rivers.items():
        points.sort(key=lambda p: p.position)

    return rivers

