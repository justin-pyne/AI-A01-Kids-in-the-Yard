from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import pandas as pd

from utils import decade, MAX_YEAR


class PersonFactory:
    """
    pandas-based loader + sampler
    """

    def __init__(
        self,
        rng: random.Random,
        life_expectancy_path: str,
        birth_marriage_path: str,
        first_names_path: str,
        gender_name_probability_path: str,
        last_names_path: str,
        rank_to_probability_path: str,
    ):
        self.rng = rng

        self.life_expectancy_by_year: Dict[int, float] = {}
        self.birth_rate_by_decade: Dict[int, float] = {}
        self.marriage_rate_by_decade: Dict[int, float] = {}


        self.first_names: Dict[Tuple[int, str], Tuple[List[str], List[float]]] = {}

        self.gendered_name_prob: Dict[Tuple[int, str], float] = {}

        self.last_names_by_decade: Dict[int, Tuple[List[str], List[float]]] = {}

        self.rank_prob: Dict[int, float] = {}

        self.load_all(
            life_expectancy_path,
            birth_marriage_path,
            first_names_path,
            gender_name_probability_path,
            last_names_path,
            rank_to_probability_path,
        )

    # normalization helpers
    def normalize_year_like(self, raw) -> int:
        """
        parses string year into int
        """
        s = str(raw).strip()
        extracted = pd.Series([s]).str.extract(r"(\d{4})", expand=False).iloc[0]
        if pd.isna(extracted):
            raise ValueError(f"Could not parse year/decade value: {raw!r}")
        return int(extracted)

    def normalize_gender(self, raw: str) -> str:
        """
        normalizing gender strings to "M" or "F"
        """
        s = str(raw).strip().lower()
        if s == "male":
            return "M"
        if s == "female":
            return "F"
        if s in {"m", "f"}:
            return s.upper()
        return str(raw).strip()

    # loaders (pandas)
    def load_all(
        self,
        life_expectancy_path: str,
        birth_marriage_path: str,
        first_names_path: str,
        gender_name_probability_path: str,
        last_names_path: str,
        rank_to_probability_path: str,
    ) -> None:
        self.load_rank_to_probability(rank_to_probability_path)
        self.load_life_expectancy(life_expectancy_path)
        self.load_birth_marriage(birth_marriage_path)
        self.load_first_names(first_names_path)
        self.load_gender_name_probability(gender_name_probability_path)
        self.load_last_names(last_names_path)

    def load_rank_to_probability(self, path: str) -> None:
        """
        applies probabilities to ranks
        """
        df = pd.read_csv(path, header=None)
        if df.empty:
            raise ValueError("rank_to_probability.csv appears empty or unreadable.")
        row0 = df.iloc[0].tolist()
        probs = [float(x) for x in row0 if pd.notna(x)]
        if not probs:
            raise ValueError("rank_to_probability.csv appears empty after parsing.")
        self.rank_prob = {i + 1: probs[i] for i in range(len(probs))}

    def load_life_expectancy(self, path: str) -> None:
        df = pd.read_csv(path)
        if "Year" not in df.columns or "Period life expectancy at birth" not in df.columns:
            raise ValueError(f"life_expectancy.csv columns unexpected: {list(df.columns)}")

        df = df[["Year", "Period life expectancy at birth"]].dropna()
        df["Year"] = df["Year"].apply(self.normalize_year_like)
        df["Period life expectancy at birth"] = df["Period life expectancy at birth"].astype(float)

        self.life_expectancy_by_year = dict(
            zip(df["Year"].tolist(), df["Period life expectancy at birth"].tolist())
        )

    def load_birth_marriage(self, path: str) -> None:
        df = pd.read_csv(path)
        expected = {"decade", "birth_rate", "marriage_rate"}
        if not expected.issubset(df.columns):
            raise ValueError(f"birth_and_marriage_rates.csv columns unexpected: {list(df.columns)}")

        df = df[list(expected)].dropna()
        df["decade"] = df["decade"].apply(self.normalize_year_like)
        df["birth_rate"] = df["birth_rate"].astype(float)
        df["marriage_rate"] = df["marriage_rate"].astype(float)

        self.birth_rate_by_decade = dict(zip(df["decade"], df["birth_rate"]))
        self.marriage_rate_by_decade = dict(zip(df["decade"], df["marriage_rate"]))

    def load_first_names(self, path: str) -> None:
        df = pd.read_csv(path)
        expected = {"decade", "gender", "name", "frequency"}
        if not expected.issubset(df.columns):
            raise ValueError(f"first_names.csv columns unexpected: {list(df.columns)}")

        df = df[list(expected)].dropna()
        df["decade"] = df["decade"].apply(self.normalize_year_like)
        df["gender"] = df["gender"].apply(self.normalize_gender)
        df["name"] = df["name"].astype(str).str.strip()
        df["frequency"] = df["frequency"].astype(float)

        grouped = df.groupby(["decade", "gender"], sort=False)
        buckets: Dict[Tuple[int, str], Tuple[List[str], List[float]]] = {}
        for (d, g), sub in grouped:
            buckets[(int(d), str(g))] = (sub["name"].tolist(), sub["frequency"].tolist())
        self.first_names = buckets

    def load_gender_name_probability(self, path: str) -> None:
        df = pd.read_csv(path)
        expected = {"decade", "gender", "probability"}
        if not expected.issubset(df.columns):
            raise ValueError(f"gender_name_probability.csv columns unexpected: {list(df.columns)}")

        df = df[list(expected)].dropna()
        df["decade"] = df["decade"].apply(self.normalize_year_like)
        df["gender"] = df["gender"].apply(self.normalize_gender)
        df["probability"] = df["probability"].astype(float)

        self.gendered_name_prob = {
            (int(row["decade"]), str(row["gender"])): float(row["probability"])
            for _, row in df.iterrows()
        }

    def load_last_names(self, path: str) -> None:
        df = pd.read_csv(path)
        expected = {"Decade", "Rank", "LastName"}
        if not expected.issubset(df.columns):
            raise ValueError(f"last_names.csv columns unexpected: {list(df.columns)}")

        df = df[list(expected)].dropna()
        df["Decade"] = df["Decade"].apply(self.normalize_year_like)
        df["Rank"] = df["Rank"].astype(int)
        df["LastName"] = df["LastName"].astype(str).str.strip()

        df["weight"] = df["Rank"].map(lambda r: float(self.rank_prob.get(int(r), 0.0)))

        buckets: Dict[int, Tuple[List[str], List[float]]] = {}
        for d, sub in df.groupby("Decade", sort=False):
            buckets[int(d)] = (sub["LastName"].tolist(), sub["weight"].tolist())
        self.last_names_by_decade = buckets

    # sampling via random.choices
    def sample_gender(self) -> str:
        return "M" if self.rng.random() < 0.5 else "F"

    def sample_year_died(self, year_born: int) -> int:
        """
        computes year of death based on life expectancy and random variation
        """
        d = decade(year_born)
        expected = self.life_expectancy_by_year.get(d)
        if expected is None:
            available = sorted(self.life_expectancy_by_year.keys())
            if not available:
                raise ValueError("No life expectancy data loaded.")
            closest = min(available, key=lambda x: abs(x - d))
            expected = self.life_expectancy_by_year[closest]

        lifespan = int(round(expected + self.rng.randint(-10, 10)))
        lifespan = max(0, lifespan)
        return year_born + lifespan

    def names_for_decade_gender(self, d: int, gender: str) -> Tuple[List[str], List[float]]:
        key = (d, gender)
        if key in self.first_names:
            return self.first_names[key]

        decades = sorted({k[0] for k in self.first_names.keys() if k[1] == gender})
        if not decades:
            raise ValueError(f"No first-name data for gender {gender}")
        closest = min(decades, key=lambda x: abs(x - d))
        return self.first_names[(closest, gender)]

    def sample_first_name(self, year_born: int, gender: str) -> str:
        """
        samples a first name based on gender and decade of birth
        """
        d = decade(year_born)
        p_gendered = self.gendered_name_prob.get((d, gender), 1.0)
        use_gendered = self.rng.random() < p_gendered

        if use_gendered:
            names, weights = self.names_for_decade_gender(d, gender)
            return self.rng.choices(names, weights=weights, k=1)[0]

        pooled_names: List[str] = []
        pooled_weights: List[float] = []
        for g in ("M", "F"):
            n, w = self.names_for_decade_gender(d, g)
            pooled_names.extend(n)
            pooled_weights.extend(w)

        return self.rng.choices(pooled_names, weights=pooled_weights, k=1)[0]

    def should_have_partner(self, year_born: int) -> bool:
        d = decade(year_born)
        p = float(self.marriage_rate_by_decade.get(d, 0.0))
        return self.rng.random() < p

    def sample_partner_birth_year(self, person_birth_year: int) -> int:
        y = person_birth_year + self.rng.randint(-10, 10)
        return max(0, min(y, MAX_YEAR))

    def sample_num_children(self, year_born: int, has_partner: bool) -> int:
        """
        computes number of children based on birth rate and random variation
        """
        d = decade(year_born)
        base = float(self.birth_rate_by_decade.get(d, 0.0))
        n = math.ceil(base + self.rng.uniform(-1.5, 1.5))
        n = max(0, n)
        if not has_partner and n > 0:
            n = max(0, n - 1)
        return n

    def child_birth_years(self, parent_a_year: int, parent_b_year: int, n: int) -> List[int]:
        """
        computes child birth years based on parent birth years and number of children
        """
        if n <= 0:
            return []

        elder = max(parent_a_year, parent_b_year)
        start = elder + 25
        end = elder + 45

        if n == 1:
            years = [start]
        else:
            step = (end - start) / (n - 1)
            years = [int(round(start + i * step)) for i in range(n)]
        years.sort()
        return years

    def sample_last_name(self, year_born: int) -> str:
        """
        samples a last name based on decade of birth and rank probability
        """
        d = decade(year_born)
        if d not in self.last_names_by_decade:
            available = sorted(self.last_names_by_decade.keys())
            if not available:
                raise ValueError("No last-name data loaded.")
            d = min(available, key=lambda x: abs(x - d))

        names, weights = self.last_names_by_decade[d]
        return self.rng.choices(names, weights=weights, k=1)[0]