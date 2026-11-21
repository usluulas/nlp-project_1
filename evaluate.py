"""Evaluate both Ollama settings against the same manually annotated samples."""

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from classify import MODEL, OPTIONS, SCHEMA, classify


CATEGORIES = tuple(SCHEMA["properties"]["category"]["enum"])
TURKISH_DOTTED_I = str.maketrans({"İ": "i", "I": "ı"})


def normalize_name(name):
    name = unicodedata.normalize("NFC", name).translate(TURKISH_DOTTED_I).lower()
    return " ".join(name.split())


def inspect_output(raw):
    """Measure direct parsing before looking for an object in a wrapped reply."""
    try:
        return True, True, json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            try:
                return False, True, json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        return False, False, None


def valid_schema(value):
    return (
        isinstance(value, dict)
        and value.keys() == SCHEMA["properties"].keys()
        and value["category"] in CATEGORIES
        and isinstance(value["entities"], list)
        and all(isinstance(name, str) for name in value["entities"])
    )


def load_gold(path):
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not samples:
        raise ValueError("The sample file is empty.")
    for sample in samples:
        names = sample["gold_entities"]
        if names is None:
            raise ValueError(f"Missing manual annotation for {sample['id']}.")
        if not isinstance(names, list) or any(
            not isinstance(name, str)
            or not normalize_name(name)
            or normalize_name(name) not in normalize_name(sample["text"])
            for name in names
        ):
            raise ValueError(f"Invalid manual annotation for {sample['id']}.")
        if sample["category"] not in CATEGORIES:
            raise ValueError(f"Unknown gold category for {sample['id']}.")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Sample IDs must be unique.")
    return samples


def score_reply(sample, raw):
    raw_ok, cleaned_ok, value = inspect_output(raw)
    schema_ok = cleaned_ok and valid_schema(value)
    gold_names = {normalize_name(name) for name in sample["gold_entities"]}
    predicted_names = {normalize_name(name) for name in value["entities"]} if schema_ok else set()
    return {
        "id": sample["id"],
        "raw": raw,
        "raw_parse_ok": raw_ok,
        "cleaned_parse_ok": cleaned_ok,
        "schema_ok": schema_ok,
        "gold_category": sample["category"],
        "predicted_category": value["category"] if schema_ok else None,
        "entity_tp": len(gold_names & predicted_names),
        "entity_fp": len(predicted_names - gold_names),
        "entity_fn": len(gold_names - predicted_names),
    }


def f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def summarize(rows):
    confusion = Counter((row["gold_category"], row["predicted_category"]) for row in rows)
    gold_counts = Counter(row["gold_category"] for row in rows)
    predicted_counts = Counter(row["predicted_category"] for row in rows)
    category_scores = []
    for label in gold_counts:
        tp = confusion[label, label]
        category_scores.append(f1(tp, predicted_counts[label] - tp, gold_counts[label] - tp))
    return {
        "samples": len(rows),
        "raw_parse_rate": sum(row["raw_parse_ok"] for row in rows) / len(rows),
        "cleaned_parse_rate": sum(row["cleaned_parse_ok"] for row in rows) / len(rows),
        "schema_validity_rate": sum(row["schema_ok"] for row in rows) / len(rows),
        "classification_accuracy": sum(confusion[label, label] for label in gold_counts) / len(rows),
        "classification_macro_f1": sum(category_scores) / len(category_scores),
        "entity_micro_f1": f1(
            sum(row["entity_tp"] for row in rows),
            sum(row["entity_fp"] for row in rows),
            sum(row["entity_fn"] for row in rows),
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--show-replies", action="store_true", help="Display raw replies as they arrive")
    args = parser.parse_args()

    try:
        samples = load_gold(args.samples)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    result = {"model": MODEL, "options": {**OPTIONS, "temperature": args.temperature}, "settings": {}}
    for label, constrained in (("prompt-only", False), ("schema-constrained", True)):
        rows = []
        for index, sample in enumerate(samples, start=1):
            if args.show_replies:
                print(f"\n{label} | temperature={args.temperature} | {sample['id']}", flush=True)
                print(sample["text"], flush=True)
            raw = classify(sample["text"], constrained=constrained, temperature=args.temperature)
            rows.append(score_reply(sample, raw))
            if args.show_replies:
                expected = {"category": sample["category"], "entities": sample["gold_entities"]}
                print(f"Raw reply:\n{raw}\nExpected: {json.dumps(expected, ensure_ascii=False)}", flush=True)
            else:
                print(f"\r{label}: {index}/{len(samples)}", end="", flush=True)
        summary = summarize(rows)
        result["settings"][label] = {"summary": summary, "rows": rows}
        print(f"\n{json.dumps(summary, indent=2)}", flush=True)

    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
