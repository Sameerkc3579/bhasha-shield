"""Localize the English seeds and generate the benign side of the guard's training data
(docs/v2-plan.md, Phase 2 steps 5-6). Runs on a free GPU (notebooks/guardset_colab.ipynb); every
stage also runs with `--fake` (no model) so CI covers the plumbing.

    python -m corpus.guardset.generate all --base-url http://localhost:8000/v1 --model <id>
    python -m corpus.guardset.generate all --fake          # CI / laptop: no model

Stages (each resumable: a request already in its cache file is never sent again):

1. `payloads`  English seed payloads (corpus/guardset/seed.py) -> hi (IndicTrans2 draft if present,
               then an LLM rewrite for natural style) -> hi_latn and mixed (LLM rewrites of the hi text).
2. `bodies`    ordinary carrier texts (emails, web pages, chats, ...) per language, from a topic list:
               the negatives, and the hosts the positives are spliced into.
3. `advbenign` adversarial-benign texts (warnings that quote an attack, recipes, notices, how-tos).
4. `assemble`  GuardItems -> corpus/guardset/drafts/generated_{lang}.jsonl, the input of `build`.

Every model output is checked before it is kept: checker tokens (amounts, ids, emails, URLs) must
survive byte for byte, hi must be Devanagari, hi_latn and mixed Latin, and no output may talk about
itself ("Here is the translation"). Rejected outputs are counted per stage, never silently patched.
Licensing: generated text only; no external corpus is used here (that is a G0 licence-table decision).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Protocol

from corpus.guardset import carriers
from corpus.guardset.schema import GuardItem
from corpus.guardset.seed import Payload, load_goals, load_templates, payloads

DRAFTS = Path("corpus/guardset/drafts")
WORK = DRAFTS / "work"
LANGS = ("en", "hi", "hi_latn", "mixed")
DEVANAGARI = re.compile("[ऀ-ॿ]")

TOKEN = re.compile(
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*"                  # emails and UPI ids
    r"|https?://\S+|www\.\S+"                         # URLs
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:in|com|co|net|dev|org)(?:/[\w./-]*)?"   # domains and links
    r"|\b[A-Z]{4}0[A-Z0-9]{6}\b"                      # IFSC
    r"|\b[A-Z]{2,}\d[A-Z0-9]*\b"                      # policy and booking numbers
    r"|\d[\d,]*\d|\b\d\b"                             # amounts, phone and account numbers
)
META = re.compile(r"^(here is|here's|translation|sure[,!]|okay[,!]|note:|i cannot|i can't|as an ai)"
                  r"|\b(the (hindi|romanized|translated) (text|version))\b", re.I)

LANG_NAME = {"hi": "Hindi in Devanagari script",
             "hi_latn": "Hindi written in Roman (Latin) script, the way people type Hindi on WhatsApp "
                        "(for example 'aapka khata', 'kripya', 'turant'), with no Devanagari",
             "mixed": "casual Hindi-English code-mixed text (Hinglish) in Latin script, the way urban "
                      "Indians text each other, with no Devanagari"}


def tokens(text: str) -> list[str]:
    return [t.rstrip(".,;:)") for t in TOKEN.findall(text)]


def problems(src: str, out: str, lang: str, *, translation: bool = True) -> list[str]:
    """Why an output may not be kept (empty list: keep it). `translation`: also compare its length
    with the source's (off for generated texts, whose source is only a topic)."""
    p = []
    out_s = out.strip()
    if not out_s:
        return ["empty"]
    missing = [t for t in tokens(src) if t not in out_s]
    if missing:
        p.append(f"lost tokens {missing[:3]}")
    has_dev = bool(DEVANAGARI.search(out_s))
    if lang == "hi" and not has_dev:
        p.append("no Devanagari")
    if lang in ("hi_latn", "mixed") and has_dev:
        p.append("Devanagari in a Latin-script language")
    if META.search(out_s.splitlines()[0]):
        p.append("talks about itself")
    if translation and not 0.4 <= len(out_s) / max(len(src), 1) <= 3.0:
        p.append("length ratio")
    return p


# ------------------------------------------------------------------------ backends

class Backend(Protocol):
    model_id: str

    def __call__(self, prompts: list[str], *, temperature: float, seed: int) -> list[str]: ...


class OpenAIBackend:
    """Any OpenAI-compatible server (vLLM on the free GPU). No paid API: base_url is local."""

    def __init__(self, base_url: str, model: str, *, workers: int = 16, max_tokens: int = 700):
        self.base_url, self.model_id, self.workers, self.max_tokens = base_url.rstrip("/"), model, workers, max_tokens

    def _one(self, prompt: str, temperature: float, seed: int) -> str:
        import urllib.request

        body = json.dumps({"model": self.model_id, "messages": [{"role": "user", "content": prompt}],
                           "temperature": temperature, "seed": seed, "max_tokens": self.max_tokens}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()

    def __call__(self, prompts, *, temperature, seed):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda p: self._one(p, temperature, seed), prompts))


class FakeBackend:
    """No model: deterministic outputs in the right script that keep every token (CI)."""

    model_id = "fake"
    MARK = {"hi": "कृपया ध्यान दें:", "hi_latn": "kripya dhyaan dein:", "mixed": "please dhyaan do:"}

    def __call__(self, prompts, *, temperature, seed):
        out = []
        for p in prompts:
            lang = re.search(r"<<lang:(\w+)>>", p).group(1)
            src = p.split("<<text>>", 1)[1].split("<</text>>", 1)[0]
            keep = " ".join(tokens(src))
            body = re.sub(r"[ऀ-ॿ]+", "", src) if lang != "hi" else src
            out.append(f"{self.MARK.get(lang, '')} {body.strip()} {keep}".strip())
        return out


# ------------------------------------------------------------------------- prompts

def _wrap(lang: str, text: str) -> str:
    return f"<<lang:{lang}>>\n<<text>>{text}<</text>>"


def prompt_localize(lang: str, english: str, draft: str | None = None) -> str:
    keep = ", ".join(dict.fromkeys(tokens(english))) or "(none)"
    if lang == "hi":
        task = ("Rewrite the machine-translated Hindi draft below so it reads like natural Hindi that a "
                "scammer in India would really write. Keep the meaning of the English original."
                if draft else "Translate the English text below into natural Hindi that a scammer in India "
                              "would really write.")
    else:
        task = (f"Rewrite the Hindi text below as {LANG_NAME[lang]}. Keep the meaning and the tone.")
    src = draft if draft else english
    return (f"{task}\nCopy these strings exactly, character for character: {keep}.\n"
            f"Keep line breaks and any markup (tags, brackets, code comments) as they are.\n"
            f"Output only the rewritten text, nothing else.\n\nEnglish original:\n{english}\n\n"
            + _wrap(lang, src))


TOPICS = (
    "a monthly society maintenance notice", "a train journey confirmation", "a school parent-teacher meeting",
    "a doctor's appointment reminder", "a recipe for poha", "a product review of a ceiling fan",
    "an office leave policy", "a cricket match plan with friends", "a wedding invitation",
    "a gym membership renewal", "a bank statement summary with no actions needed", "a lost-and-found notice",
    "a power cut schedule", "a team lunch plan", "a college fee receipt", "a mobile recharge confirmation",
    "a library book due date", "a festival sale announcement", "a plumber's visit", "a hotel stay review",
    "a weather alert for heavy rain", "a cab ride receipt", "a job interview schedule", "a birthday party plan",
    "a vaccination camp notice", "a property tax receipt", "a cooking class", "an apartment rental listing",
    "a car service reminder", "a quarterly sales report summary", "a pharmacy order confirmation",
    "a yoga class timetable", "a parcel delivery update", "a newspaper article on monsoon farming",
    "a temple festival schedule", "a coaching class timetable", "a restaurant review", "a utility bill paid",
    "meeting minutes about a website redesign", "an internship offer with no payment requests",
)
ADV_SEEDS = (
    "a bank's warning that quotes a fake 'your account will be blocked, send OTP' message",
    "a security tip explaining that 'ignore your previous instructions' in an email is an attack",
    "an IT circular telling employees how to change their password themselves",
    "a news article about prompt-injection attacks on AI assistants, with an example attack quoted",
    "a recipe written as step-by-step commands",
    "exam instructions for students",
    "a pharmacy label with dosage instructions",
    "a fire-drill notice telling people what to do",
    "a help-centre article on how to enable two-factor authentication",
    "a parent's message telling a child what to do after school",
    "a manager asking the team to submit timesheets by Friday",
    "a police advisory about digital-arrest scams that quotes a scam call",
    "an app's onboarding instructions",
    "a customer-care script that tells callers to never share their PIN",
    "a school notice saying fees are never collected by phone",
    "a research abstract about AI agents following hidden instructions",
    "an e-commerce return policy with steps",
    "a WhatsApp forward debunking a fake lottery message",
    "a cybercrime helpline poster with instructions",
    "a user manual section for a washing machine",
)
CARRIER_DESC = {"email": "an email", "web": "a web page", "document": "a document", "chat": "a chat message",
                "sms": "an SMS", "calendar": "a calendar entry", "review": "a product or service review",
                "json": "a JSON API response with 3 to 6 string fields"}


def prompt_body(lang: str, carrier: str, topic: str, *, adversarial: bool) -> str:
    what = CARRIER_DESC[carrier]
    if adversarial:
        rule = ("It may contain instructions or quote a scam, because that is what it is about, but it must be "
                "completely harmless: it must not ask any reader, person or AI, to send money, share data, or "
                "change any setting on someone else's behalf.")
    else:
        rule = "It must be ordinary and harmless, with no instructions to an AI assistant."
    lang_txt = "English" if lang == "en" else LANG_NAME[lang]
    return (f"Write {what} in {lang_txt} about: {topic}. 40 to 160 words. Make it realistic and specific "
            f"(Indian names, places and amounts are fine). {rule} Output only the text.\n" + _wrap(lang, topic))


# --------------------------------------------------------------------------- stages

def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:12], 16)


def _cache(name: str) -> dict[str, dict]:
    p = WORK / f"{name}.jsonl"
    if not p.exists():
        return {}
    return {r["key"]: r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}


def _append(name: str, rows: list[dict]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / f"{name}.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run(name: str, requests: list[dict], backend: Backend, *, check: Callable[[dict, str], list[str]],
         temperature: float, batch: int = 64, retries: int = 2) -> dict[str, dict]:
    """Send every request not yet cached; keep outputs that pass `check`, retry the rest with a new
    seed; after the last retry a failing output is stored with its reasons (counted, never used)."""
    done = _cache(name)
    todo = [r for r in requests if r["key"] not in done]
    for attempt in range(retries + 1):
        if not todo:
            break
        failed = []
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            outs = backend([r["prompt"] for r in chunk], temperature=temperature, seed=attempt)
            rows = []
            for r, o in zip(chunk, outs):
                why = check(r, o)
                if why and attempt < retries:
                    failed.append(r)
                    continue
                rows.append({**{k: v for k, v in r.items() if k != "prompt"}, "text": o.strip(), "rejected": why,
                             "model": backend.model_id, "attempt": attempt,
                             "prompt_sha": hashlib.sha256(r["prompt"].encode()).hexdigest()[:12]})
            _append(name, rows)
        todo = failed
    return _cache(name)


def stage_payloads(backend: Backend, *, drafts_hi: dict[str, str] | None = None) -> dict[str, dict]:
    """English payloads -> hi -> hi_latn, mixed. `drafts_hi`: seed_id -> IndicTrans2 draft, if any."""
    rows = payloads(load_goals(), load_templates())
    hi = _run("payloads_hi", [{"key": p.seed_id, "seed_id": p.seed_id, "lang": "hi", "en": p.text,
                               "draft": (drafts_hi or {}).get(p.seed_id),
                               "prompt": prompt_localize("hi", p.text, (drafts_hi or {}).get(p.seed_id))}
                              for p in rows], backend, check=lambda r, o: problems(r["en"], o, "hi"), temperature=0.3)
    ok = {k: v for k, v in hi.items() if not v["rejected"]}
    out = {f"hi|{k}": v for k, v in hi.items()}
    for lang in ("hi_latn", "mixed"):
        res = _run(f"payloads_{lang}", [{"key": k, "seed_id": k, "lang": lang, "en": v["en"], "hi": v["text"],
                                         "prompt": prompt_localize(lang, v["en"], v["text"])} for k, v in ok.items()],
                   backend, check=lambda r, o, lang=lang: problems(r["en"], o, lang), temperature=0.4)
        out |= {f"{lang}|{k}": v for k, v in res.items()}
    return out


def stage_bodies(backend: Backend, *, adversarial: bool, per_topic: int = 2) -> dict[str, dict]:
    name = "advbenign" if adversarial else "bodies"
    seeds = ADV_SEEDS if adversarial else TOPICS
    reqs = []
    for lang in LANGS:
        for t_i, topic in enumerate(seeds):
            for v in range(per_topic * (2 if adversarial else 1)):
                carrier = list(CARRIER_DESC)[_h(name, lang, t_i, v) % len(CARRIER_DESC)]
                key = f"{lang}|{t_i}|{v}"
                reqs.append({"key": key, "lang": lang, "topic": topic, "topic_id": t_i, "variant": v,
                             "carrier": carrier, "prompt": prompt_body(lang, carrier, topic, adversarial=adversarial)})
    return _run(name, reqs, backend, check=lambda r, o: problems("", o, r["lang"], translation=False),
                temperature=0.9)


def splice(host: dict, payload: str, *, position: str, seed: int) -> str:
    """Put a payload into a host text; a JSON host gets it in one of its string fields."""
    if host["carrier"] == "json":
        try:
            obj = json.loads(host["text"])
        except ValueError:
            obj = None
        if isinstance(obj, dict) and obj:
            keys = sorted(k for k, v in obj.items() if isinstance(v, str)) or ["notes"]
            k = keys[seed % len(keys)]
            obj[k] = (str(obj.get(k, "")) + " " + payload).strip()
            return json.dumps(obj, ensure_ascii=False)
        return carriers.wrap("json", host["text"], payload, seed=seed)
    return carriers.insert(host["text"], payload, position=position, seed=seed)


def _payload_text(lang: str, p: Payload, loc: dict[str, dict]) -> str | None:
    if lang == "en":
        return p.text
    r = loc.get(f"{lang}|{p.seed_id}")
    return r["text"] if r and not r["rejected"] else None


def assemble(loc: dict[str, dict], bodies: dict[str, dict], adv: dict[str, dict], *,
             embed_per_payload: int = 1, bare_share: float = 0.5) -> dict[str, list[GuardItem]]:
    """Positives: every localized payload spliced into `embed_per_payload` bodies of its language (and
    a bare copy for a hashed `bare_share`); negatives: every body and every adversarial-benign text."""
    seeds = payloads(load_goals(), load_templates())
    out: dict[str, list[GuardItem]] = {lang: [] for lang in LANGS}
    for lang in LANGS:
        hosts = sorted((r for r in bodies.values() if r["lang"] == lang and not r["rejected"]), key=lambda r: r["key"])
        for r in hosts:
            out[lang].append(GuardItem.new(text=r["text"], label="benign", lang=lang, family="ordinary",
                                           group=f"topic:{r['topic_id']}", carrier=r["carrier"],
                                           source=f"generated:{r['model']}", provenance={"key": r["key"]}))
        for r in sorted((r for r in adv.values() if r["lang"] == lang and not r["rejected"]), key=lambda r: r["key"]):
            out[lang].append(GuardItem.new(text=r["text"], label="benign", lang=lang, family="adversarial_benign",
                                           group=f"adv:{r['topic_id']}", carrier=r["carrier"], adversarial_benign=True,
                                           source=f"generated:{r['model']}", provenance={"key": r["key"]}))
        for p in seeds:
            text = _payload_text(lang, p, loc)
            if text is None:
                continue
            src = "seed" if lang == "en" else f"generated:{loc[f'{lang}|{p.seed_id}']['model']}"
            prov = {"seed_id": p.seed_id}
            if _h("bare", lang, p.seed_id) % 1000 < bare_share * 1000:
                out[lang].append(GuardItem.new(text=text, label="injection", lang=lang, family=p.family, goal=p.goal,
                                               carrier="none", source=src, provenance=prov))
            for k in range(embed_per_payload if hosts else 0):
                host = hosts[_h("host", lang, p.seed_id, k) % len(hosts)]
                pos = ("start", "middle", "end", "random")[_h("pos", lang, p.seed_id, k) % 4]
                text_in = splice(host, text, position=pos, seed=_h("seed", p.seed_id, k) % 2**31)
                out[lang].append(GuardItem.new(text=text_in, label="injection", lang=lang, family=p.family,
                                               goal=p.goal, carrier=host["carrier"], source=src,
                                               provenance={**prov, "host": host["key"], "position": pos}))
    return out


def report(loc: dict, bodies: dict, adv: dict, items: dict[str, list[GuardItem]]) -> dict:
    def rej(d):
        by: dict[str, dict[str, int]] = {}
        for k, r in d.items():
            lang = r.get("lang", k.split("|", 1)[0])
            s = by.setdefault(lang, {"kept": 0, "rejected": 0})
            s["rejected" if r["rejected"] else "kept"] += 1
        return dict(sorted(by.items()))
    return {"payloads": rej(loc), "bodies": rej(bodies), "advbenign": rej(adv),
            "items": {lang: {"injection": sum(i.label == "injection" for i in v),
                             "benign": sum(i.label == "benign" for i in v)} for lang, v in items.items()}}


def main(argv: list[str] | None = None) -> int:
    global WORK
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["payloads", "bodies", "advbenign", "assemble", "all"])
    ap.add_argument("--fake", action="store_true", help="no model: CI and laptop")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--drafts-hi", type=Path, help="IndicTrans2 drafts: JSONL of {seed_id, text}")
    ap.add_argument("--work", type=Path, default=WORK)
    a = ap.parse_args(argv)
    WORK = a.work

    backend: Backend = FakeBackend() if a.fake else OpenAIBackend(a.base_url, a.model)
    drafts = None
    if a.drafts_hi and a.drafts_hi.exists():
        drafts = {r["seed_id"]: r["text"] for r in map(json.loads, a.drafts_hi.read_text(encoding="utf-8").splitlines())
                  if r.get("text")}
    loc = stage_payloads(backend, drafts_hi=drafts) if a.stage in ("payloads", "all") else None
    bod = stage_bodies(backend, adversarial=False) if a.stage in ("bodies", "all") else None
    adv = stage_bodies(backend, adversarial=True) if a.stage in ("advbenign", "all") else None
    if a.stage in ("assemble", "all"):
        loc = loc if loc is not None else {f"{n.split('_', 1)[1]}|{k}": v
                                           for n in ("payloads_hi", "payloads_hi_latn", "payloads_mixed")
                                           for k, v in _cache(n).items()}
        bod, adv = bod if bod is not None else _cache("bodies"), adv if adv is not None else _cache("advbenign")
        items = assemble(loc, bod, adv)
        DRAFTS.mkdir(parents=True, exist_ok=True)
        for lang, rows in items.items():
            (DRAFTS / f"generated_{lang}.jsonl").write_bytes(
                "".join(r.model_dump_json() + "\n" for r in rows).encode("utf-8"))
        rep = report(loc, bod, adv, items)
        (WORK / "generate_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
