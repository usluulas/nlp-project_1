"""Compare two raw Ollama replies to the same article at temperature zero."""

import json
from pathlib import Path

from classify import classify


def main():
    demo_path = Path(__file__).resolve().parent / "data" / "demo.json"
    samples = json.loads(demo_path.read_text(encoding="utf-8"))
    article = samples[0]["text"]

    print("Run 1 (temperature 0.0):", flush=True)
    first_reply = classify(article, constrained=True, temperature=0.0)
    print(first_reply, flush=True)

    print("\nRun 2 (temperature 0.0):", flush=True)
    second_reply = classify(article, constrained=True, temperature=0.0)
    print(second_reply, flush=True)

    print("\nIdentical:", first_reply == second_reply)


if __name__ == "__main__":
    main()
