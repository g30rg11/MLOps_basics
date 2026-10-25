import csv
from pathlib import Path

from main import amount_of_substance


def main() -> None:
    output_path = Path("data/results.csv")

    with (
        Path("data/substances.csv").open(encoding="utf-8-sig", newline="") as source,
        output_path.open("w", encoding="utf-8", newline="") as target,
    ):
        reader = csv.DictReader(source)
        writer = csv.writer(target)
        writer.writerow(["substance", "amount_mol"])

        for row in reader:
            amount = amount_of_substance(
                float(row["mass_g"]),
                float(row["molar_mass_g_mol"]),
            )
            writer.writerow([row["substance"], amount])

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
