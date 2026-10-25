import csv
import json
from pathlib import Path


def main() -> None:
    with Path("data/results.csv").open(encoding="utf-8", newline="") as source:
        amounts = [float(row["amount_mol"]) for row in csv.DictReader(source)]

    report = {
        "samples_count": len(amounts),
        "total_amount_mol": round(sum(amounts), 6),
    }

    output_path = Path("reports/summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
