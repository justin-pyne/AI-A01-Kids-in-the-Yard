from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from person import Person
from person_factory import PersonFactory
from utils import MAX_YEAR, decade


class FamilyTree:
    """
    Generates a family tree starting from two people in 1950
    Continues generating descendants and married in partners until no more children are
    produced or until MAX_YEAR.
    """

    def __init__(self, factory: PersonFactory):
        self.factory = factory
        self.people: Dict[int, Person] = {}
        self._next_id = 1
        self.root_last_names: Tuple[str, str] = ("", "")

    # Helpers
    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def add_person(
        self,
        year_born: int,
        *,
        gender: Optional[str] = None,
        last_name: Optional[str] = None,
        parent_ids: Tuple[int, ...] = tuple(),
        is_direct_descendant: bool = False,
    ) -> Person:
        g = gender or self.factory.sample_gender()
        first = self.factory.sample_first_name(year_born, g)
        died = self.factory.sample_year_died(year_born)
        ln = last_name if last_name is not None else self.factory.sample_last_name(year_born)

        p = Person(
            id=self._new_id(),
            gender=g,
            first_name=first,
            last_name=ln,
            year_born=year_born,
            year_died=died,
            parent_ids=parent_ids,
            is_direct_descendant=is_direct_descendant,
        )
        self.people[p.id] = p
        return p

    def link_partners(self, a: Person, b: Person) -> None:
        a.partner_id = b.id
        b.partner_id = a.id

    # Generation
    def generate(self) -> None:
        """
        Build the full simulated population.
        Starts with two roots born in 1950, assigns them as partners, then
        grows the tree using a queue until MAX_YEAR.
        """
        rng = self.factory.rng

        # choosing last name for 1950 roots
        root1_last = self.factory.sample_last_name(1950)
        root2_last = self.factory.sample_last_name(1950)
        self.root_last_names = (root1_last, root2_last)

        # creating starting two people
        r1 = self.add_person(1950, last_name=root1_last, is_direct_descendant=True)
        r2 = self.add_person(1950, last_name=root2_last, is_direct_descendant=True)
        self.link_partners(r1, r2)

        queue: List[int] = [r1.id, r2.id]

        # marking processed units to prevent duplicate child generation
        processed_units = set()

        while queue:
            pid = queue.pop(0)
            person = self.people[pid]

            if person.year_born >= MAX_YEAR:
                continue

            partner: Optional[Person] = self.people.get(person.partner_id) if person.partner_id else None

            # creating a partner if needed
            if partner is None and self.factory.should_have_partner(person.year_born):
                partner_year = self.factory.sample_partner_birth_year(person.year_born)
                partner_last = self.factory.sample_last_name(partner_year)
                partner_gender = "F" if person.gender == "M" else "M"

                partner = self.add_person(
                    partner_year,
                    gender=partner_gender,
                    last_name=partner_last,
                    is_direct_descendant=False,
                )
                self.link_partners(person, partner)
                queue.append(partner.id)

            # computing a "parenting unit" key and only processing once
            if partner is not None:
                unit_key = tuple(sorted((person.id, partner.id)))
            else:
                unit_key = (person.id,)

            if unit_key in processed_units:
                continue
            processed_units.add(unit_key)

            # deciding number of children
            has_partner = partner is not None
            n_children = self.factory.sample_num_children(person.year_born, has_partner=has_partner)
            if n_children <= 0:
                continue

            other_year = partner.year_born if partner else person.year_born
            child_years = self.factory.child_birth_years(person.year_born, other_year, n_children)

            direct_line = person.is_direct_descendant or (partner.is_direct_descendant if partner else False)

            for cy in child_years:
                if cy > MAX_YEAR:
                    continue

                # picking last name from parents
                if direct_line:
                    child_last = rng.choice(list(self.root_last_names))
                    child_direct = True
                else:
                    child_last = self.factory.sample_last_name(cy)
                    child_direct = False

                parent_ids = (person.id,) + ((partner.id,) if partner else tuple())

                child = self.add_person(
                    cy,
                    last_name=child_last,
                    parent_ids=parent_ids,
                    is_direct_descendant=child_direct,
                )

                # updating child links
                person.children_ids.append(child.id)
                if partner:
                    partner.children_ids.append(child.id)

                # enqueuing child
                queue.append(child.id)

    # CLI queries
    def total_people(self) -> int:
        return len(self.people)

    def total_by_decade(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for p in self.people.values():
            d = decade(p.year_born)
            counts[d] = counts.get(d, 0) + 1
        return dict(sorted(counts.items()))

    def duplicate_names(self) -> Dict[str, int]:
        """
        Returns a dict of full_name -> count for names that appear more than once
        """
        freq: Dict[str, int] = {}
        for p in self.people.values():
            name = p.full_name
            freq[name] = freq.get(name, 0) + 1
        return {name: c for name, c in freq.items() if c > 1}

    def get_person(self, pid: int) -> Person:
        return self.people[pid]