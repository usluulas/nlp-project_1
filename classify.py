"""Send the same news classification prompt to Ollama in either JSON mode."""

import argparse
import json
from urllib.request import Request, urlopen

from prepare_data import CATEGORIES


MODEL = "llama3.2:3b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OPTIONS = {"seed": 42, "num_ctx": 2048, "num_predict": 256, "top_k": 40, "top_p": 0.9}
SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES.values())},
        "entities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["category", "entities"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "Analyze a Turkish news excerpt. Assign its main topic to one category and "
    "extract the person names in the excerpt. Copy the names as they appear in the text. "
    "If a person is mentioned only by surname, extract the surname. If the same person's "
    "full name appears, include only the full name. Exclude titles, forms of address, "
    "organizations, and locations. Remove inflectional suffixes separated from a name "
    "by a space or an apostrophe. List each name once. Return an empty list if there "
    "are no person names. Return only a JSON object that conforms to the schema. "
    "Do not include Markdown or explanations. JSON schema: " + json.dumps(SCHEMA, ensure_ascii=False)
)


def classify(text, *, constrained, temperature):
    """Return raw model output so evaluation can inspect it before cleaning."""
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "options": {**OPTIONS, "temperature": temperature},
    }
    if constrained:
        payload["format"] = SCHEMA

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)["message"]["content"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="A single Turkish news excerpt")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    for label, constrained in (("prompt-only", False), ("schema-constrained", True)):
        print(f"\n{label}", flush=True)
        print(classify(args.text, constrained=constrained, temperature=args.temperature))
