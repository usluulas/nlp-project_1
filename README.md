# Structured Extraction from Turkish News with a Local LLM

This NLP project uses Llama 3.2 3B through local Ollama to classify Turkish news excerpts and extract the names of people. It compares prompt-only generation with schema-constrained generation on the same 105 manually annotated excerpts from TTC-4900.

[Technical report](technical%20report/Technical%20Report.pdf) · [Main experiment results and raw replies](test-en-tr-t0.json)

## Main results

Each condition ran on the same 105 excerpts at temperature `0.0`.

| Metric | Prompt-only | Schema-constrained |
| --- | ---: | ---: |
| Raw JSON parse rate | 0.00% | 100.00% |
| JSON parse rate after cleaning | 100.00% | 100.00% |
| Schema validity after cleaning | 100.00% | 100.00% |
| Category accuracy | 42.86% (45/105) | 44.76% (47/105) |
| Category macro-F1 | 40.47% | 41.83% |
| Person-name micro-F1 | 40.86% | 38.10% |

All prompt-only replies arrived inside Markdown code fences. The cleaner recovered a valid JSON object from each reply. The schema-constrained replies were directly parseable. The schema also led to two more correct category predictions overall, while person-name micro-F1 fell by 2.76 percentage points.

## Setup and reproduction

The experiment used Python `3.14.7`, Ollama `0.40.16` and `llama3.2:3b` with `Q4_K_M` quantization and local model ID `a80c4f17acd5`. The scripts use the Python standard library and send requests to the local Ollama server at `127.0.0.1:11434`. [Ollama model page](https://ollama.com/library/llama3.2:3b)

1. Open the Ollama desktop app. On a command-line installation, run `ollama serve` in a separate terminal.
2. If the model is missing, run `ollama pull llama3.2:3b`.
3. From the repository root, run the main experiment with a new output filename.

```powershell
python -X utf8 evaluate.py data/test.json reproduced-main.json --temperature 0.0
```

On Windows, `uv run --offline python` can also be used in place of `python`. The evaluator stores the model, settings, summary metrics and raw replies in the output JSON file. Each run requires a new output filename.

| Generation option | Main run |
| --- | ---: |
| `temperature` | `0.0` |
| `seed` | `42` |
| `num_ctx` | `2048` |
| `num_predict` | `256` |
| `top_k` | `40` |
| `top_p` | `0.9` |

## Data and evaluation

The test set has 15 excerpts from each of seven categories. The labels are `dünya`, `ekonomi`, `kültür`, `sağlık`, `siyaset`, `spor` and `teknoloji`. The preparation script replaces underscores with spaces, keeps up to the first 50 words of each article, removes duplicate excerpts and samples by category with seed `42`. [TTC-4900 dataset card](https://huggingface.co/datasets/savasy/ttc4900)

The category labels come from the dataset. Person names were annotated by hand using the same excerpts sent to the model. The annotations use the fullest name present in the excerpt, or a surname if it is the only form. Titles, places, organizations and separated inflectional suffixes are excluded. The evaluator uses exact name matching after Unicode NFC normalization, Turkish-aware lowercasing (`İ` → `i` and `I` → `ı`), and whitespace normalization.

Both conditions use the same English instructions and Turkish input. Prompt-only mode includes the schema in the prompt. Schema-constrained mode also sends it in Ollama’s `format` field. The schema requires a category from the seven labels and a list of person names. The evaluator tries `json.loads()` on the raw reply, retries on the text between the first and last braces if needed, and checks the resulting object against the schema. Failed validation counts as a category error and an empty person-name prediction.

The repository includes the annotated test set in [data/test.json](data/test.json). To rebuild the same sample selection, place the [source CSV](https://raw.githubusercontent.com/savasy/TurkishTextClassification/master/TTC4900.csv) in the repository root and run `python prepare_data.py`. The script checks an existing test file against the generated selection and preserves its manual annotations. The source CSV used for this experiment has SHA-256 `e17b79e89a3679ed77b3d5fd6d855fca43e9986a714cd4927c646c2be692c23e`.

## Demo and repeatability check

[data/demo.json](data/demo.json) contains five fictional articles outside TTC-4900. Each command runs both settings and shows the article, raw reply and expected answer.

```powershell
python -X utf8 evaluate.py data/demo.json reproduced-demo-t0.json --temperature 0.0 --show-replies
python -X utf8 evaluate.py data/demo.json reproduced-demo-t1.json --temperature 1.0 --show-replies
python -X utf8 defense_c2.py
```

In the demo, direct JSON parsing succeeded on `0/5` prompt-only replies and `5/5` schema-constrained replies at both temperatures. Both conditions got `2/5` categories right at each temperature. The separate repeatability check sends the same article twice at temperature zero and compares the two raw replies.

[Demo results at 0.0](demo-en-tr-t0.json) · [Demo results at 1.0](demo-en-tr-t1.json)

In VS Code, open **Run and Debug** and select either demo or the main experiment from the dropdown. The [.vscode/launch.json](.vscode/launch.json) configurations show the output in **Debug Console**.

## Project files

| File | Purpose |
| --- | --- |
| [prepare_data.py](prepare_data.py) | Selects reproducible test excerpts from TTC-4900. |
| [classify.py](classify.py) | Builds the prompt and makes the local Ollama request. |
| [evaluate.py](evaluate.py) | Parses, validates and scores responses in both settings. |
| [defense_c2.py](defense_c2.py) | Compares two replies to the same prompt at temperature zero. |
