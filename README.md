# option-order-check

Does your multiple-choice LLM eval depend on where the answers are placed?

`option-order-check` shows every question to your model once per cyclic rotation of its options (A-B-C-D, B-C-D-A, ...) and reports, with uncertainty:

- mean accuracy over all rotations (95% Wilson interval)
- the **flip rate**: the share of questions whose chosen answer changes with option order
- accuracy by the slot the correct answer sits in (position bias shows up here)
- the share of questions answered correctly under every ordering

Pure standard library, no dependencies. Works offline with a built-in mock model, or with any OpenAI-compatible chat endpoint (OpenAI, Ollama, vLLM, LM Studio).

## Install and try it (no API key)

```bash
pip install option-order-check
option-order-check --demo --provider mock
```

The mock model has a deliberate lean toward one answer slot, so you can see what a biased model looks like in the output.

## Run it on your own model

Your file is JSONL, one question per line, with the correct option as a 0-based index:

```json
{"question": "Which gas do plants absorb?", "options": ["Oxygen", "Carbon dioxide", "Helium", "Neon"], "answer": 1}
```

```bash
# OpenAI
export OPENAI_API_KEY=...
option-order-check mytests.jsonl --provider openai --model gpt-4o-mini

# A local model through any OpenAI-compatible server (Ollama shown)
option-order-check mytests.jsonl --provider openai --model llama3 \
    --base-url http://localhost:11434/v1 --api-key-env NONE
```

Add `--json-out report.json` to save the numbers.

## Reading the output

A high flip rate at decent accuracy means your score depends on where answers are placed. A large spread between the best and worst correct-answer slot means the model has a position preference. See the audit this tool came from: [Shuffle the answers and small models change their minds](https://robert-morong-research.netlify.app/writing/arc-option-order/).

## Limits

The Wilson interval treats the rotations of one question as independent, which they are not, so read it as a rough guide. The flip-rate interval resamples questions. This tool measures option-order sensitivity only. It is not a full reliability audit, and it says nothing about safety, bias or domain correctness.

## License

MIT. Part of the free tooling at https://robert-morong-research.netlify.app/free/
