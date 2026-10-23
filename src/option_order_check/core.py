#!/usr/bin/env python3
"""option_order_check.py - how much does your multiple-choice eval depend on option order?

Each question is shown once per cyclic rotation of its options (A-B-C-D, B-C-D-A, ...).
The script reports, with uncertainty:
  * mean accuracy over all rotations (Wilson 95% interval)
  * the flip rate: the share of questions whose chosen answer changes with option order
  * accuracy by the slot the correct answer sits in (position bias shows up here)
  * correct-under-all-rotations rate

Pure standard library. Works offline with --provider mock, or with any OpenAI-compatible
chat endpoint (OpenAI, Ollama, vLLM, LM Studio) with --provider openai.

Input: a JSONL file, one question per line:
  {"question": "...", "options": ["...", "...", "...", "..."], "answer": 2}
where "answer" is the 0-based index of the correct option.

Examples:
  python option_order_check.py demo.jsonl --provider mock
  python option_order_check.py mytests.jsonl --provider openai --model gpt-4o-mini \
      --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY
  python option_order_check.py mytests.jsonl --provider openai --model llama3 \
      --base-url http://localhost:11434/v1 --api-key-env NONE

License: MIT. Part of the free tooling from https://robert-morong-research.netlify.app/free/
"""
from __future__ import annotations
import argparse, json, math, os, random, re, sys, urllib.request

LETTERS = "ABCDEFGH"


def rotate(seq, r):
    n = len(seq)
    return [seq[(i + r) % n] for i in range(n)]


def prompt(question, options):
    lines = [question.strip(), ""] + [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    lines += ["", "Answer with the letter of the correct option only."]
    return "\n".join(lines)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def bootstrap_ci(values, reps=2000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    stats = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return (stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1])


class Mock:
    """Offline stand-in with a deliberate slot bias, so you can see what a biased model looks like."""
    def __init__(self, bias=0.35, skill=0.55, seed=0):
        self.rng = random.Random(seed)
        self.bias, self.skill = bias, skill

    def choose(self, question, options, correct_display):
        n = len(options)
        if self.rng.random() < self.skill:
            return correct_display
        if self.rng.random() < self.bias:
            return min(1, n - 1)  # leans toward slot B
        return self.rng.randrange(n)


class OpenAICompatible:
    def __init__(self, model, base_url, api_key, temperature=0.0, timeout=60):
        self.model, self.url = model, base_url.rstrip("/") + "/chat/completions"
        self.key, self.temperature, self.timeout = api_key, temperature, timeout

    def choose(self, question, options, correct_display):
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": 8,
                "messages": [{"role": "user", "content": prompt(question, options)}]}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            text = json.load(r)["choices"][0]["message"]["content"]
        m = re.search(r"\b([A-H])\b", text.strip().upper())
        if not m or LETTERS.index(m.group(1)) >= len(options):
            return -1  # unparseable answer counts as wrong and as its own "pick"
        return LETTERS.index(m.group(1))


def load(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            opts = o["options"]
            if not (2 <= len(opts) <= len(LETTERS)) or not (0 <= o["answer"] < len(opts)):
                raise SystemExit(f"line {ln}: need 2-{len(LETTERS)} options and a valid 0-based answer index")
            items.append(o)
    if not items:
        raise SystemExit("no questions found")
    return items


def run(items, model):
    """Return per-question list of (picked underlying option index or -1) over rotations."""
    results = []
    for it in items:
        n = len(it["options"])
        picks = []
        for r in range(n):
            shown = rotate(it["options"], r)
            correct_display = (it["answer"] - r) % n
            d = model.choose(it["question"], shown, correct_display)
            picks.append(-1 if d < 0 else (d + r) % n)  # map back to the underlying option
        results.append(picks)
    return results


def report(items, results):
    n_q = len(items)
    rots = len(items[0]["options"])
    total = sum(len(p) for p in results)
    correct = sum(1 for it, p in zip(items, results) for x in p if x == it["answer"])
    lo, hi = wilson(correct, total)
    flips = [1 if len(set(p)) > 1 else 0 for p in results]
    all_ok = [1 if all(x == it["answer"] for x in p) else 0 for it, p in zip(items, results)]
    flo, fhi = bootstrap_ci(flips)
    by_slot = {}
    for it, p in zip(items, results):
        n = len(it["options"])
        for r, x in enumerate(p):
            slot = (it["answer"] - r) % n
            c, t = by_slot.get(slot, (0, 0))
            by_slot[slot] = (c + (x == it["answer"]), t + 1)
    out = {
        "questions": n_q, "predictions": total,
        "accuracy_mean": correct / total, "accuracy_ci95_wilson": [lo, hi],
        "flip_rate": sum(flips) / n_q, "flip_rate_ci95_bootstrap": [flo, fhi],
        "correct_under_all_orders": sum(all_ok) / n_q,
        "accuracy_by_correct_slot": {LETTERS[s]: c / t for s, (c, t) in sorted(by_slot.items())},
        "note": "Predictions on the same question are not independent; the Wilson interval treats them as if they were, so read it as a rough guide. The flip-rate bootstrap resamples questions.",
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("testset")
    ap.add_argument("--provider", choices=["mock", "openai"], default="mock")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the key; use NONE for local servers")
    ap.add_argument("--json-out", help="also write the report as JSON")
    a = ap.parse_args()
    items = load(a.testset)
    if a.provider == "mock":
        model = Mock()
    else:
        key = None if a.api_key_env == "NONE" else os.environ.get(a.api_key_env)
        if a.api_key_env != "NONE" and not key:
            raise SystemExit(f"set {a.api_key_env} or pass --api-key-env NONE for a local server")
        model = OpenAICompatible(a.model, a.base_url, key)
    rep = report(items, run(items, model))
    pc = lambda x: f"{100 * x:.1f}%"
    print(f"\nQuestions: {rep['questions']}   Predictions: {rep['predictions']}")
    print(f"Mean accuracy over all orderings: {pc(rep['accuracy_mean'])}  (95% Wilson {pc(rep['accuracy_ci95_wilson'][0])} to {pc(rep['accuracy_ci95_wilson'][1])})")
    print(f"Answer flips with option order:   {pc(rep['flip_rate'])}  (95% bootstrap {pc(rep['flip_rate_ci95_bootstrap'][0])} to {pc(rep['flip_rate_ci95_bootstrap'][1])})")
    print(f"Correct under every ordering:     {pc(rep['correct_under_all_orders'])}")
    print("Accuracy by slot of the correct answer: " + "  ".join(f"{k}={pc(v)}" for k, v in rep["accuracy_by_correct_slot"].items()))
    spread = max(rep["accuracy_by_correct_slot"].values()) - min(rep["accuracy_by_correct_slot"].values())
    print(f"Slot spread (max-min): {100 * spread:.1f} points. Large spreads mean the score depends on where the answer is shown.")
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(rep, f, indent=2)
        print("wrote", a.json_out)


if __name__ == "__main__":
    main()
