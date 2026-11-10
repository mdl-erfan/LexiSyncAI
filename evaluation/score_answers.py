"""Step 9.4: turn the manual labels into the final numbers per question type."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
ANSWERS = RESULTS / "answers_final.json"
LABELS = RESULTS / "labels.json"
OUT = RESULTS / "scores.json"
TYPES = ["factual", "multihop", "unanswerable"]

def check_labels(rows: list[dict], labels: dict) -> None:
    """Stop early on a missing or misspelled label, so a typo never becomes a number."""
    for r in rows:
        if r["abstained"]:
            continue
        label = labels.get(r["id"])
        if label is None:
            raise ValueError(f"{r['id']}: no label")
        if label["answer"] not in {"correct", "partly", "wrong"}:
            raise ValueError(f"{r['id']}: answer label {label['answer']!r}")
        for n, verdict in label["citations"].items():
            if verdict not in {"yes", "no"}:
                raise ValueError(f"{r['id']} citation [{n}]: {verdict!r}")

def score(rows: list[dict], labels: dict) -> dict:
    """Counts (not ratios) for one group of questions; ratios are computed when printing."""
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    answered = [r for r in answerable if not r["abstained"]]
    verdicts = [labels[r["id"]]["answer"] for r in answered]
    cites = [v for r in answered for v in labels[r["id"]]["citations"].values()]
    latency = [r["latency_s"] for r in rows]
    return {
        "n": len(rows),
        "answerable": len(answerable),
        "correct": verdicts.count("correct"),
        "partly": verdicts.count("partly"),
        "wrong": verdicts.count("wrong"),
        "wrongly_refused": len(answerable) - len(answered),
        "citations": len(cites),
        "citations_supported": cites.count("yes"),
        "unanswerable": len(unanswerable),
        "unanswerable_refused": sum(r["abstained"] for r in unanswerable),
        "latency_mean": round(statistics.mean(latency), 1),
        "latency_median": round(statistics.median(latency), 1),
        "latency_max": round(max(latency), 1),
    }

def ratio(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.2f})" if whole else "-"


def print_table(scores: dict) -> None:
    """Markdown table, ready to paste into the README."""
    print("| Type | n | Correct | Partly | Wrong | Citation precision "
          "| Unanswerable refused | Answerable wrongly refused | Latency mean / median / max (s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, s in scores.items():
        print(f"| {name} | {s['n']} | {ratio(s['correct'], s['answerable'])} "
              f"| {s['partly'] if s['answerable'] else '-'} | {s['wrong'] if s['answerable'] else '-'} "
              f"| {ratio(s['citations_supported'], s['citations'])} "
              f"| {ratio(s['unanswerable_refused'], s['unanswerable'])} "
              f"| {ratio(s['wrongly_refused'], s['answerable'])} "
              f"| {s['latency_mean']} / {s['latency_median']} / {s['latency_max']} |")


def main() -> None:
    data = json.loads(ANSWERS.read_text())
    rows = data["rows"]
    labels = json.loads(LABELS.read_text())
    check_labels(rows, labels)

    scores = {t: score([r for r in rows if r["type"] == t], labels) for t in TYPES}
    scores["all"] = score(rows, labels)
    print_table(scores)

    OUT.write_text(json.dumps({"meta": data["meta"], "scores": scores}, indent=2))
    print(f"\nSaved to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()