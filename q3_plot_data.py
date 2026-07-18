from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULT_PATH = PROJECT_DIR / "outputs" / "model" / "q3_results.json"
OUTPUT_PATH = (
    PROJECT_DIR / "paper" / "figures" / "q3_tmax_curves.dat"
)


def main() -> None:
    results = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    typical = results["appendix_typical_scenario"]["curve"]
    scenario_map = {
        (row["scenario"], row["band_state"]): row["curve"]
        for row in results["combined_scenarios"]
    }
    a_curve = scenario_map[("A", "without_band")]
    b_no_band = scenario_map[("B", "without_band")]
    b_band = scenario_map[
        ("B", "with_effective_band_conditional")
    ]
    rows = [
        "e_ratio typical A B_no_band B_effective_band",
    ]
    for index, typical_row in enumerate(typical):
        rows.append(
            " ".join(
                [
                    f"{float(typical_row['eccentricity_ratio']):.10f}",
                    f"{float(typical_row['Tmax_Nm']):.10f}",
                    f"{float(a_curve[index]['combined_Tmax_Nm']):.10f}",
                    f"{float(b_no_band[index]['combined_Tmax_Nm']):.10f}",
                    f"{float(b_band[index]['combined_Tmax_Nm']):.10f}",
                ]
            )
        )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
