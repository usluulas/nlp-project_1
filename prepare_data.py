"""Create the reproducible test sample from the TTC-4900 news dataset."""

import csv
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "TTC4900.csv"
OUTPUT = ROOT / "data"
SEED = 42
CATEGORIES = {
    "dunya": "dünya",
    "ekonomi": "ekonomi",
    "kultur": "kültür",
    "saglik": "sağlık",
    "siyaset": "siyaset",
    "spor": "spor",
    "teknoloji": "teknoloji",
}
MAX_WORDS = 50
MIN_WORDS = 25
RESERVED_PER_CATEGORY = 2
TEST_PER_CATEGORY = 15


def sample_news():
    """Restore spaces in collocations and sample distinct excerpts by category."""
    groups = {label: [] for label in CATEGORIES.values()}
    seen_excerpts = set()
    with DATASET.open(encoding="utf-8-sig", newline="") as source:
        for index, row in enumerate(csv.DictReader(source), start=1):
            label = CATEGORIES[row["category"].strip()]
            words = row["text"].replace("_", " ").split()
            excerpt = " ".join(words[:MAX_WORDS])
            if len(words) < MIN_WORDS or excerpt in seen_excerpts:
                continue
            seen_excerpts.add(excerpt)
            groups[label].append({"id": f"ttc4900:{index}", "category": label, "text": excerpt})

    rng = random.Random(SEED)
    reserved, test = [], []
    for group in groups.values():
        selected = rng.sample(group, RESERVED_PER_CATEGORY + TEST_PER_CATEGORY)
        reserved.extend(selected[:RESERVED_PER_CATEGORY])
        test.extend(selected[RESERVED_PER_CATEGORY:])
    # Keep the original RNG sequence so the existing test file keeps its order.
    rng.shuffle(reserved)
    rng.shuffle(test)
    return test


def write_samples(path, samples):
    """Create the annotation file once and preserve later manual labels."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_samples = [
            {key: sample[key] for key in ("id", "category", "text")}
            for sample in existing
        ]
        if existing_samples != samples:
            raise ValueError(f"Existing samples differ from the current selection: {path}")
        print(f"{path.name}: kept {len(samples)} samples and their annotations")
        return

    records = [{**sample, "gold_entities": None} for sample in samples]
    contents = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(contents)
    print(f"{path.name}: wrote {len(records)} samples")


if __name__ == "__main__":
    test = sample_news()
    OUTPUT.mkdir(exist_ok=True)
    write_samples(OUTPUT / "test.json", test)
