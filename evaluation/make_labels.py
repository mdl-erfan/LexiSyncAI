"""Step 9.3: build a readable review sheet and an empty labels file from the final answer run."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
ANSWERS = RESULTS / "answers_final.json"
LABELS = RESULTS / "labels.json"
REVIEW = RESULTS / "review.md"

def label_template(rows: list[dict]) -> dict:
    """One entry per answered question: a blank answer label and a blank per citation."""
    labels = {}
    for r in rows:
        if r["abstained"]:
            continue  # refusals are scored automatically in 9.4
        labels[r["id"]] = {
            "answer": "",  # correct / partly / wrong
            "citations": {str(c["n"]): "" for c in r["citations"]},  # yes / no
            "note": "",    # why, for the honest-failures section
        }
    return labels

def review_sheet(rows: list[dict]) -> str:
    """Question, expected answer, system answer and every cited passage, as Markdown."""
    lines = ["# Step 9 review sheet", ""]
    for r in rows:
        lines += [f"## {r['id']} ({r['type']}, {r['badge']})", "",
                  f"**Question:** {r['question']}", "",
                  f"**Expected:** {r['expected']}", "",
                  f"**Answer:** {r['answer']}", ""]
        for c in r["citations"]:
            lines += [f"**[{c['n']}] {c['label']}**", "",
                      f"> {c['text'].replace(chr(10), ' ')}", ""]
    return "\n".join(lines)


def main() -> None:
    rows = json.loads(ANSWERS.read_text())["rows"]
    REVIEW.write_text(review_sheet(rows))
    print(f"Wrote {REVIEW.name}")
    if LABELS.exists():
        print(f"{LABELS.name} already exists - left untouched")
    else:
        LABELS.write_text(json.dumps(label_template(rows), indent=2))
        print(f"Wrote {LABELS.name}")


if __name__ == "__main__":
    main()