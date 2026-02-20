import random

from person_factory import PersonFactory
from family_tree import FamilyTree


def run_cli(tree: FamilyTree) -> None:
    prompt = (
        "Are you interested in:\n"
        "(T)otal number of people in the tree\n"
        "Total number of people in the tree by (D)ecade\n"
        "(N)ames duplicated\n"
        "(Q)uit"
    )

    while True:
        print(prompt)
        cmd = input("> ").strip().upper()

        if cmd == "Q":
            return

        if cmd == "T":
            print(f"The tree contains {tree.total_people()} people total")

        elif cmd == "D":
            counts = tree.total_by_decade()
            for d, c in counts.items():
                print(f"{d}: {c}")

        elif cmd == "N":
            dupes = tree.duplicate_names()
            num_dupe_names = len(dupes)

            if num_dupe_names == 0:
                print("There are 0 duplicate names in the tree:")
            elif num_dupe_names == 1:
                print("There is 1 duplicate name in the tree:")
            else:
                print(f"There are {num_dupe_names} duplicate names in the tree:")

            for name in sorted(dupes.keys()):
                print(f"* {name}")

        else:
            print("Invalid option.")


def main() -> None:
    print("Reading files...")

    rng = random.Random(42)

    factory = PersonFactory(
        rng=rng,
        life_expectancy_path="life_expectancy.csv",
        birth_marriage_path="birth_and_marriage_rates.csv",
        first_names_path="first_names.csv",
        gender_name_probability_path="gender_name_probability.csv",
        last_names_path="last_names.csv",
        rank_to_probability_path="rank_to_probability.csv",
    )

    tree = FamilyTree(factory)

    print("Generating family tree...")
    tree.generate()

    run_cli(tree)


if __name__ == "__main__":
    main()