from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class Person:
    id: int
    gender: str
    first_name: str
    last_name: str
    year_born: int
    year_died: int

    partner_id: Optional[int] = None
    parent_ids: Tuple[int, ...] = field(default_factory=tuple)
    children_ids: List[int] = field(default_factory=list)

    is_direct_descendant: bool = False

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"