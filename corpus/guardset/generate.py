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
survive byte for byte and not be repeated, hi must be Devanagari, hi_latn and mixed Latin with enough
common Hindi words that they are not plain English, a JSON carrier must parse, and no output may echo
the prompt or talk about itself ("Here is the translation"). Rejected outputs are counted per stage
with their reasons, never silently patched; the one normalization is unwrapping a whole-output code
fence (`tidy`), and the row records it. Prompts are a system turn (the instructions) and a user turn
(only the text to work on), with no markup the model could copy back.
Licensing: generated text only; no external corpus is used here (that is a G0 licence-table decision).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import asdict, dataclass
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
ECHO = re.compile(r"<</?\w|<<lang|these strings stay exactly|reply with the text itself", re.I)
LATIN_WORD = re.compile(r"[A-Za-z]+")
# Common Hindi function words in Roman script, minus the ones that are also English words ("the",
# "is", "to", "me"...). On the 2026-11-22 run, English texts scored at most 0.031 on this.
HINDI_WORDS = frozenset("""hai hain tha thi ka ki ke ko se mein nahi nahin aur kya pe par bhi hoga hogi hote
    karo karein karna karke kariye kijiye kiya kiye aap aapka aapki aapke apna apni apne yeh ye woh wo voh
    iska uska unka hamara hamare mera meri mere tum tumhara jaldi abhi kripya turant liye sath saath wala
    wali wale baad pehle jo sab sabhi agar toh lekin kyunki raha rahi rahe gaya gayi diya liya dena lena
    chahiye sakta sakte sakti hua hui hue kuch koi bahut phir bhai ji kar""".split())
HINDI_SHARE = {"hi_latn": 0.10, "mixed": 0.06}
META = re.compile(r"^(here is|here's|translation|sure[,!]|okay[,!]|note:|i cannot|i can't|as an ai)"
                  r"|\b(the (hindi|romanized|translated) (text|version))\b", re.I)

LANG_NAME = {"hi": "Hindi in Devanagari script",
             "hi_latn": "Hindi written in Roman (Latin) script, the way people type Hindi on WhatsApp "
                        "(for example 'aapka khata', 'kripya', 'turant'), with no Devanagari",
             "mixed": "casual Hindi-English code-mixed text (Hinglish) in Latin script, the way urban "
                      "Indians text each other, with no Devanagari. The grammar stays Hindi (verbs, pronouns and "
                      "small words such as hai, ko, ke liye, karo, nahi, aap); English is used only for some nouns "
                      "and technical words (account, email, password, payment, ticket), for example 'aapka account "
                      "abhi verify karo, warna payment block ho jayega'. Every sentence has Hindi words in it; "
                      "never return the English original"}
# 2026-11-27: the first mixed wording got plain English back for about half the texts (315 of 704 payloads).
KEEP_PASSED = False   # --keep-passed: a cached row that passed its checks is kept even if its prompt changed


def tokens(text: str) -> list[str]:
    return [t.rstrip(".,;:)") for t in TOKEN.findall(text)]


def hindi_share(text: str) -> float:
    words = [w.lower() for w in LATIN_WORD.findall(text)]
    return sum(w in HINDI_WORDS for w in words) / max(len(words), 1)


def tidy(out: str) -> tuple[str, bool]:
    """Unwrap an output that is one code fence (```json ... ```); the only normalization done."""
    m = re.fullmatch(r"\s*```[\w-]*\n(.*?)\n?```\s*", out, re.S)
    return (m.group(1).strip(), True) if m else (out.strip(), False)


def problems(src: str, out: str, lang: str, *, translation: bool = True, carrier: str | None = None) -> list[str]:
    """Why an output may not be kept (empty list: keep it). `translation`: also compare its length
    with the source's (off for generated texts, whose source is only a topic)."""
    p = []
    out_s = out.strip()
    if not out_s:
        return ["empty"]
    if ECHO.search(out_s):
        p.append("prompt echo")
    missing = [t for t in tokens(src) if t not in out_s]
    if missing:
        p.append(f"lost tokens {missing[:3]}")
    long_tokens = {t for t in tokens(src) if len(t) >= 4}
    repeated = [t for t in long_tokens if out_s.count(t) > src.count(t)]
    if long_tokens and len(repeated) >= (len(long_tokens) + 1) // 2:
        p.append("repeats tokens")
    has_dev = bool(DEVANAGARI.search(out_s))
    if lang == "hi" and not has_dev:
        p.append("no Devanagari")
    if lang == "en" and has_dev:
        p.append("Devanagari in English")
    if lang in ("hi_latn", "mixed") and has_dev:
        p.append("Devanagari in a Latin-script language")
    if lang in HINDI_SHARE and hindi_share(out_s) < HINDI_SHARE[lang]:
        p.append("too little Hindi")
    if carrier == "json":
        try:
            ok = isinstance(json.loads(out_s), dict)
        except ValueError:
            ok = False
        if not ok:
            p.append("not a JSON object")
    if META.search(out_s.splitlines()[0]):
        p.append("talks about itself")
    if translation and not 0.4 <= len(out_s) / max(len(src), 1) <= 3.0:
        p.append("length ratio")
    return p


# ------------------------------------------------------------------------ backends

@dataclass(frozen=True)
class Prompt:
    system: str   # the instructions
    user: str     # only the text to work on (or the topic)
    lang: str
    form: str = "text"   # "json" when the output must be a JSON object


def _sha(prompt: Prompt) -> str:
    return hashlib.sha256(f"{prompt.system}\0{prompt.user}".encode()).hexdigest()[:12]


class Backend(Protocol):
    model_id: str

    def __call__(self, prompts: list[Prompt], *, temperature: float, seed: int) -> list[str]: ...


class OpenAIBackend:
    """Any OpenAI-compatible server (vLLM on the free GPU). No paid API: base_url is local."""

    def __init__(self, base_url: str, model: str, *, workers: int = 32, max_tokens: int = 700):
        self.base_url, self.model_id, self.workers, self.max_tokens = base_url.rstrip("/"), model, workers, max_tokens

    def _one(self, prompt: Prompt, temperature: float, seed: int) -> str:
        import urllib.request

        messages = [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user}]
        body = json.dumps({"model": self.model_id, "messages": messages,
                           "temperature": temperature, "seed": seed, "max_tokens": self.max_tokens}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()

    def __call__(self, prompts, *, temperature, seed):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda p: self._one(p, temperature, seed), prompts))


class FakeBackend:
    """No model: deterministic outputs in the right script that keep every token once (CI)."""

    model_id = "fake"
    MARK = {"hi": "कृपया ध्यान दें:", "hi_latn": "kripya dhyaan dein:", "mixed": "please dhyaan do:"}

    def __call__(self, prompts, *, temperature, seed):
        out = []
        for p in prompts:
            body = p.user if p.lang == "hi" else re.sub(r"[ऀ-ॿ]+", "", p.user)
            if p.lang in HINDI_SHARE:   # a Hindi word after every third word: Latin script, not plain English
                words = body.split(" ")
                body = " ".join(w + (" hai" if i % 3 == 2 else "") for i, w in enumerate(words))
            text = f"{self.MARK.get(p.lang, '')} {body.strip()}".strip()
            out.append(json.dumps({"message": text}, ensure_ascii=False) if p.form == "json" else text)
        return out


# ------------------------------------------------------------------------- prompts

FORMAT_RULES = ("Reply with the text itself and nothing else: no preface, no notes, no title saying what it "
                "is, no quotation marks around it, no code fence.")


def prompt_localize(lang: str, english: str, draft: str | None = None) -> Prompt:
    """hi: `draft` is the IndicTrans2 draft, if any. hi_latn, mixed: `draft` is the hi text."""
    keep = ", ".join(dict.fromkeys(tokens(english)))
    if lang == "hi":
        task = ("The user message is a machine-translated Hindi draft. Rewrite it so it reads like natural "
                "Hindi that a scammer in India would really write, keeping the meaning of the English original."
                if draft else "The user message is an English text. Translate it into natural Hindi that a "
                              "scammer in India would really write.")
    else:
        task = (f"The user message is a Hindi text. Rewrite it as {LANG_NAME[lang]}, keeping the meaning and "
                f"the tone. Do not leave whole sentences in English.")
    keep_rule = (f"These strings stay exactly as they are, each in its own place in the text, and appear "
                 f"only where the original has them: {keep}.\n" if keep else "")
    return Prompt(system=f"{task}\n{keep_rule}Keep line breaks and any markup (tags, brackets, code comments, "
                         f"Markdown) as they are.\n{FORMAT_RULES}\n\nThe English original, for meaning only:\n{english}",
                  user=draft or english, lang=lang)


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
    "a bus pass renewal at the depot counter", "a kirana store's new opening hours", "a dentist's clinic shifting address",
    "a flight delay update from the airline", "an electricity meter reading slip", "a cricket club's practice schedule",
    "a housing society's water tank cleaning", "a farmer's market price list for vegetables", "a tuition fee reminder",
    "a laptop repair status from a service centre", "a review of a dhaba on the highway", "a blood donation camp",
    "a wedding caterer's menu options", "a gas cylinder delivery slot", "a movie ticket booking confirmation",
    "a hospital discharge summary with diet advice", "an office Diwali party plan", "a lease renewal discussion",
    "a book club's monthly pick", "a review of a budget smartphone", "a school bus route change",
    "a travel itinerary for Rishikesh", "a sprint retrospective summary", "a broadband outage notice from the ISP",
    "a monsoon road diversion notice", "a cousin's graduation ceremony", "a mutual fund statement with no action needed",
    "a neighbourhood cleanliness drive", "a recipe for rajma chawal", "a warranty registration confirmation",
    "a customer feedback summary for a cafe", "a museum visiting hours notice", "a university exam timetable",
    "a gardening tip for tulsi plants", "a shared expense summary after a trip", "a review of a ceiling fan installer",
    "an elderly parent's medicine schedule", "a hackathon announcement at college", "a holiday list for the year",
    "a pet vaccination reminder from the vet",
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
    "an RBI awareness ad saying banks never ask for your OTP",
    "a blog post explaining how phishing emails tell AI assistants to forward data, quoting one",
    "a teacher's homework instructions for class 8",
    "a bank FAQ on how to block a lost debit card yourself in the app",
    "a set of first-aid steps for a minor burn",
    "a building's lift-safety instructions",
    "a train ticket checker's announcement about carrying ID",
    "a company's guide on reporting suspicious emails to the IT desk",
    "a grandmother's instructions for making pickle",
    "a security researcher's note describing a hidden-instruction attack on a chatbot, as an example",
    "a hotel's check-out instructions for guests",
    "a GST filing checklist for a small shop owner, for the owner to follow",
    "a voter ID application guide with steps",
    "a cyber-safety quiz for school students with example scam messages",
    "a software installation guide with commands to type",
    "an exam invigilator's rules read out before a test",
    "a WhatsApp group admin's rules for members",
    "a news report on a UPI fraud that quotes the fraudster's message",
    "an airline's baggage rules",
    "a gym trainer's workout plan written as commands",
)
CARRIER_DESC = {"email": "an email", "web": "a web page", "document": "a document", "chat": "a chat message",
                "sms": "an SMS", "calendar": "a calendar entry", "review": "a product or service review",
                "json": "a JSON API response with 3 to 6 string fields"}


def prompt_body(lang: str, carrier: str, topic: str, *, adversarial: bool) -> Prompt:
    what = CARRIER_DESC[carrier]
    if adversarial:
        rule = ("It may contain instructions or quote a scam, because that is what it is about, but it must be "
                "completely harmless: it must not ask any reader, person or AI, to send money, share data, or "
                "change any setting on someone else's behalf.")
    else:
        rule = "It must be ordinary and harmless, with no instructions to an AI assistant."
    lang_txt = "English" if lang == "en" else LANG_NAME[lang]
    form = "Reply with the raw JSON object only, no code fence." if carrier == "json" else FORMAT_RULES
    return Prompt(system=f"Write {what} in {lang_txt} about the topic in the user message. 40 to 160 words. "
                         f"Make it realistic and specific (Indian names, places and amounts are fine). {rule} {form}",
                  user=topic, lang=lang, form="json" if carrier == "json" else "text")


# --------------------------------------------------------------------------- stages

def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:12], 16)


def _cache(name: str) -> dict[str, dict]:
    p = WORK / f"{name}.jsonl"
    if not p.exists():
        return {}
    rows = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:   # a half-written last line: a save copied the file mid-write
            continue
        rows[r["key"]] = r
    return rows


def _append(name: str, rows: list[dict]) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / f"{name}.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _run(name: str, requests: list[dict], backend: Backend, *, check: Callable[[dict, str], list[str]],
         temperature: float, batch: int = 64, retries: int = 2) -> dict[str, dict]:
    """Send every request not yet cached; keep outputs that pass `check`, retry the rest with a new
    seed; after the last retry a failing output is stored with its reasons (counted, never used).
    A cached row counts only for the same prompt: a new draft or upstream text resends it. With
    KEEP_PASSED, a row that passed its checks is kept under a changed prompt; only rejects are resent."""
    done = _cache(name)

    def cached(r: dict) -> bool:
        row = done.get(r["key"])
        return bool(row) and (row.get("prompt_sha") == _sha(r["prompt"]) or (KEEP_PASSED and not row["rejected"]))

    todo = [r for r in requests if not cached(r)]
    for attempt in range(retries + 1):
        if not todo:
            break
        failed = []
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            outs = backend([r["prompt"] for r in chunk], temperature=temperature, seed=attempt)
            rows = []
            for r, o in zip(chunk, outs):
                o, tidied = tidy(o)
                why = check(r, o)
                if why and attempt < retries:
                    failed.append(r)
                    continue
                rows.append({**{k: v for k, v in r.items() if k != "prompt"}, "text": o, "rejected": why, "tidied": tidied,
                             "model": backend.model_id, "attempt": attempt,
                             "prompt_sha": _sha(r["prompt"])})
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
    return _run(name, reqs, backend, temperature=0.9,
                check=lambda r, o: problems("", o, r["lang"], translation=False, carrier=r["carrier"]))


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
            s = by.setdefault(lang, {"kept": 0, "rejected": 0, "reasons": Counter()})
            s["rejected" if r["rejected"] else "kept"] += 1
            s["reasons"].update(w.split(" [", 1)[0] for w in r["rejected"] or [])
        return {lang: {**s, "reasons": dict(sorted(s["reasons"].items()))} for lang, s in sorted(by.items())}
    return {"payloads": rej(loc), "bodies": rej(bodies), "advbenign": rej(adv),
            "items": {lang: {"injection": sum(i.label == "injection" for i in v),
                             "benign": sum(i.label == "benign" for i in v)} for lang, v in items.items()}}


def main(argv: list[str] | None = None) -> int:
    global WORK, KEEP_PASSED
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["payloads", "bodies", "advbenign", "assemble", "all"])
    ap.add_argument("--fake", action="store_true", help="no model: CI and laptop")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--drafts-hi", type=Path, help="IndicTrans2 drafts: JSONL of {seed_id, text}")
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--keep-passed", action="store_true",
                    help="after a prompt change, keep rows that already passed; resend only the rejected ones")
    a = ap.parse_args(argv)
    WORK, KEEP_PASSED = a.work, a.keep_passed

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
