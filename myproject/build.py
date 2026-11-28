#!/usr/bin/env python3
"""build.py - build a project step by step on free-tier LLM APIs.

The files on disk are the memory. A plain script (this one) runs the loop;
models only ever see the plan, one milestone and the few files it needs.

  python3 build.py plan  "describe the project"   # writes PLAN.md + state, stops for your review
  python3 build.py run                              # builds every pending milestone
  python3 build.py run --push                       # ... then git push
  python3 build.py status

Run it from an empty folder (the project folder). Needs only the Python
standard library. Keys come from environment variables (see ENV below).
"""
import json, os, re, subprocess, sys, time, urllib.request, urllib.error, datetime

# ----------------------------------------------------------------- config
# ENV: GEMINI_API_KEYS=k1,k2,k3  (or GEMINI_API_KEY)  GEMINI_MODEL
#      GROQ_API_KEY GROQ_MODEL   NVIDIA_API_KEY NVIDIA_MODEL
#      CLOUDFLARE_API_TOKEN CLOUDFLARE_ACCOUNT_ID CLOUDFLARE_MODEL
# Optional: BUILD_PLAN_ORDER / BUILD_WRITE_ORDER / BUILD_FIX_ORDER / BUILD_REVIEW_ORDER
#           (comma lists of provider names), BUILD_MAX_FIX (3), BUILD_GEMINI_RPM (10)
STATE_DIR = ".builder"
STATE_FILE = os.path.join(STATE_DIR, "state.json")
LOG_FILE = os.path.join(STATE_DIR, "build.log")
MAX_FILE_CHARS = 12000      # per file shown to a model
MAX_CONTEXT_CHARS = 30000   # total file text per call
MAX_ERR_CHARS = 4000
TIMEOUT = 180


def env(name, default=""):
    return os.environ.get(name, default).strip()


def order(var, default):
    return [p for p in env(var, default).split(",") if p.strip()]


ROLE_ORDER = {
    "plan": lambda: order("BUILD_PLAN_ORDER", "gemini,cloudflare,groq"),
    "write": lambda: order("BUILD_WRITE_ORDER", "gemini,cloudflare"),
    "fix": lambda: order("BUILD_FIX_ORDER", "gemini,cloudflare,groq"),
    "review": lambda: order("BUILD_REVIEW_ORDER", "nvidia,groq,gemini"),
}


def providers():
    """name -> dict(url, model, keys, rpm, tpm)"""
    gem_keys = [k.strip() for k in (env("GEMINI_API_KEYS") or env("GEMINI_API_KEY")).split(",") if k.strip()]
    p = {}
    if gem_keys:
        p["gemini"] = dict(url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                           model=env("GEMINI_MODEL", "gemini-3.5-flash-lite"), keys=gem_keys,
                           rpm=int(env("BUILD_GEMINI_RPM", "10")), tpm=0)
    if env("GROQ_API_KEY"):
        p["groq"] = dict(url="https://api.groq.com/openai/v1/chat/completions",
                         model=env("GROQ_MODEL", "openai/gpt-oss-120b"), keys=[env("GROQ_API_KEY")],
                         rpm=20, tpm=6000)
    if env("NVIDIA_API_KEY"):
        p["nvidia"] = dict(url="https://integrate.api.nvidia.com/v1/chat/completions",
                           model=env("NVIDIA_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
                           keys=[env("NVIDIA_API_KEY")], rpm=10, tpm=0)
    if env("CLOUDFLARE_API_TOKEN") and env("CLOUDFLARE_ACCOUNT_ID"):
        p["cloudflare"] = dict(
            url="https://api.cloudflare.com/client/v4/accounts/%s/ai/v1/chat/completions" % env("CLOUDFLARE_ACCOUNT_ID"),
            model=env("CLOUDFLARE_MODEL", "@cf/qwen/qwen3.8-27b"), keys=[env("CLOUDFLARE_API_TOKEN")],
            rpm=20, tpm=0)
    return p


# ----------------------------------------------------------------- logging
def log(msg):
    line = "%s %s" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ----------------------------------------------------------------- rate limiting
class KeySlot:
    """One API key: spaces calls out, tracks cool-downs and daily exhaustion."""

    def __init__(self, provider, key, rpm, tpm):
        self.provider, self.key = provider, key
        self.interval = 60.0 / rpm if rpm else 0
        self.tpm = tpm
        self.next_ok = 0.0          # earliest time for next call
        self.dead_until = 0.0       # cool-down after 429 / daily quota
        self.tok_window = []        # (time, tokens)

    def wait_time(self, est_tokens=0):
        now = time.time()
        w = max(self.next_ok, self.dead_until) - now
        if self.tpm:
            self.tok_window = [(t, n) for t, n in self.tok_window if now - t < 60]
            used = sum(n for _, n in self.tok_window)
            if used + est_tokens > self.tpm and self.tok_window:
                w = max(w, 60 - (now - self.tok_window[0][0]) + 1)
        return max(0.0, w)

    def used(self, tokens):
        now = time.time()
        self.next_ok = now + self.interval
        if self.tpm:
            self.tok_window.append((now, tokens))

    def cool(self, seconds):
        self.dead_until = time.time() + seconds


SLOTS = {}


def slots_for(name, prov):
    if name not in SLOTS:
        SLOTS[name] = [KeySlot(name, k, prov["rpm"], prov["tpm"]) for k in prov["keys"]]
    return SLOTS[name]


class Transient(Exception):
    pass


def seconds_to_utc_midnight():
    now = datetime.datetime.now(datetime.timezone.utc)
    nxt = (now + datetime.timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
    return (nxt - now).total_seconds()


def http_chat(prov, key, messages, max_tokens):
    body = json.dumps({"model": prov["model"], "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.2}).encode()
    req = urllib.request.Request(prov["url"], data=body, method="POST", headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")[:600]
        e.body_text = txt
        e.retry_after = e.headers.get("Retry-After")
        raise
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise Transient("unexpected response: %s" % str(data)[:200])


def call_model(role, messages, max_tokens=8000):
    """Try providers for this role in order; rotate keys; wait for limits. Returns text."""
    provs = providers()
    names = [n for n in ROLE_ORDER[role]() if n in provs]
    if not names:
        sys.exit("No provider configured for role '%s'. Check your environment variables." % role)
    est = sum(len(m["content"]) for m in messages) // 3 + max_tokens // 4
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        # pick the (provider, slot) that can go soonest, preferring earlier providers
        best = None
        for rank, n in enumerate(names):
            for s in slots_for(n, provs[n]):
                w = s.wait_time(est)
                if w >= 24 * 3600:
                    continue
                score = (w > 0, w if w > 0 else 0, rank)
                if best is None or score < best[0]:
                    best = (score, n, s, w)
        if best is None:
            sys.exit("All providers are out of quota for today. Run again after the reset (state is saved).")
        _, n, slot, w = best
        if w > 0:
            if w > 120:
                log("waiting %ds for %s (all keys busy)" % (w, n))
            time.sleep(min(w, 60))
            continue
        try:
            text = http_chat(provs[n], slot.key, messages, max_tokens)
            slot.used(est)
            return text
        except urllib.error.HTTPError as e:
            slot.used(0)
            body = getattr(e, "body_text", "")
            if e.code == 429:
                daily = re.search(r"per ?day|daily|neurons|4006|quota", body, re.I)
                if daily:
                    slot.cool(seconds_to_utc_midnight())
                    log("%s key ...%s: daily quota used up" % (n, slot.key[-4:]))
                else:
                    try:
                        ra = float(e.retry_after)
                    except (TypeError, ValueError):
                        ra = 30
                    slot.cool(max(ra, 10))
                    log("%s key ...%s: rate limited, backing off %ds" % (n, slot.key[-4:], ra))
            elif e.code in (500, 502, 503, 504, 408):
                slot.cool(20)
                log("%s HTTP %s, retrying" % (n, e.code))
            elif e.code == 413:
                slot.cool(60)
                log("%s HTTP 413 (too big for its limit)" % n)
            elif e.code in (401, 403):
                slot.cool(48 * 3600)
                log("%s key ...%s rejected (HTTP %s) - skipping it" % (n, slot.key[-4:], e.code))
            else:
                slot.cool(30)
                log("%s HTTP %s: %s" % (n, e.code, body[:200]))
        except (urllib.error.URLError, TimeoutError, Transient, OSError) as e:
            slot.used(0)
            slot.cool(15)
            log("%s network/response problem: %s" % (n, str(e)[:150]))
    sys.exit("Gave up waiting for a provider.")


# ----------------------------------------------------------------- state / files
def load_state():
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, STATE_FILE)


def safe_path(rel):
    rel = rel.strip().replace("\\", "/")
    if not rel or rel.startswith("/") or ".." in rel.split("/") or rel.startswith(".git/") \
            or rel.split("/")[0] in (".git", STATE_DIR) or re.match(r"^[A-Za-z]:", rel):
        return None
    if os.path.basename(rel) in (".env", ".env.local") or rel.endswith((".pem", ".key")):
        return None
    return rel


FILE_RE = re.compile(r"^=== FILE: (.+?) ===\n(.*?)\n=== END ===", re.S | re.M)


def parse_files(text):
    out = {}
    for m in FILE_RE.finditer(text):
        rel = safe_path(m.group(1))
        body = m.group(2)
        body = re.sub(r"^```[\w+-]*\n", "", body)
        body = re.sub(r"\n```\s*$", "", body)
        if rel:
            out[rel] = body + "\n"
        else:
            log("refused unsafe path from model: %r" % m.group(1))
    return out


def write_files(files):
    for rel, body in files.items():
        os.makedirs(os.path.dirname(rel) or ".", exist_ok=True)
        with open(rel, "w") as f:
            f.write(body)


def read_file(rel, cap=MAX_FILE_CHARS):
    try:
        with open(rel, errors="replace") as f:
            t = f.read()
    except OSError:
        return None
    return t if len(t) <= cap else t[:cap] + "\n...[truncated]"


def project_tree():
    out = []
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in (".git", STATE_DIR, "__pycache__", "node_modules", ".venv", "target")]
        for fn in files:
            out.append(os.path.join(root, fn)[2:])
    return sorted(out)[:200]


def context_files(rels):
    parts, total = [], 0
    for rel in rels:
        t = read_file(rel)
        if t is None:
            continue
        if total + len(t) > MAX_CONTEXT_CHARS:
            parts.append("--- %s (omitted: context budget) ---" % rel)
            continue
        total += len(t)
        parts.append("--- %s ---\n%s" % (rel, t))
    return "\n\n".join(parts) if parts else "(no files yet)"


def sh(cmd, timeout=300):
    """Run a command with a stripped environment: no API keys are visible to generated code."""
    safe_env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp"),
                "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=safe_env)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT after %ss" % timeout


def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True, text=True)


# ----------------------------------------------------------------- prompts
SYS_WRITER = ("You are a careful senior engineer. Write complete, working, secure code. "
              "Never hard-code secrets; read them from environment variables. Validate all external input. "
              "Use parameterised queries, no shell string-building from user input, no eval/exec on input. "
              "Output ONLY files in this exact format, nothing else:\n"
              "=== FILE: relative/path.ext ===\n<full file content>\n=== END ===\n"
              "Always output the FULL content of every file you touch, never partial snippets.")


def extract_json(text, need=None):
    """Find the first JSON object in text (ignores <think> blocks and prose around it).
    If `need` is given, the object must contain that key."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and (need is None or need in obj):
            return obj
    raise ValueError("no JSON object found")


def cmd_plan(idea):
    if os.path.exists(STATE_FILE):
        sys.exit("A plan already exists here (.builder/state.json). Use a fresh folder or delete .builder.")
    prompt = ("Project idea:\n%s\n\nChoose the best-fitting language/stack (prefer the simplest that does the job "
              "well). Break the work into 3-10 SMALL milestones, each producing 1-3 files that can be tested. "
              "The first milestone must set up the project skeleton and a test command that runs offline. "
              "Reply with ONLY JSON:\n"
              '{"name":"...","language":"...","summary":"...","test_cmd":"one shell command that runs ALL tests and exits non-zero on failure",'
              '"milestones":[{"id":1,"title":"...","goal":"what must work","files":["path1","path2"],"tests":"how it is verified"}]}'
              ) % idea
    text = call_model("plan", [{"role": "system", "content": "You are a pragmatic software architect."},
                               {"role": "user", "content": prompt}], 6000)
    try:
        plan = extract_json(text, "milestones")
        assert plan["milestones"] and plan["test_cmd"]
    except Exception as e:
        sys.exit("Planner returned something unusable (%s). Run plan again." % e)
    for i, m in enumerate(plan["milestones"], 1):
        m["id"] = i
        m["files"] = [f for f in (safe_path(x) for x in m.get("files", [])) if f]
        m["status"] = "pending"
        m["attempts"] = 0
    plan["idea"] = idea
    plan["approved"] = False
    save_state(plan)
    with open("PLAN.md", "w") as f:
        f.write("# %s\n\n%s\n\nLanguage: %s\nTest command: `%s`\n\n## Milestones\n" % (
            plan.get("name", "project"), plan.get("summary", ""), plan.get("language", "?"), plan["test_cmd"]))
        for m in plan["milestones"]:
            f.write("\n### %d. %s\n%s\nFiles: %s\nVerified by: %s\n" % (
                m["id"], m["title"], m["goal"], ", ".join(m["files"]), m.get("tests", "")))
    print("\nPlan written to PLAN.md. READ IT and edit PLAN.md/.builder/state.json if you want changes.")
    print("Then run:  python3 build.py run")


def cmd_status():
    st = load_state()
    print("%s  (%s)" % (st.get("name"), st.get("language")))
    for m in st["milestones"]:
        print("  [%s] %d. %s (attempts %d)" % (m["status"], m["id"], m["title"], m["attempts"]))


def ensure_git():
    if not os.path.isdir(".git"):
        git("init", "-q")
        with open(".gitignore", "a") as f:
            f.write("\n.builder/\n.env\n__pycache__/\nnode_modules/\n.venv/\n")
        git("add", "-A")
        git("-c", "user.name=builder", "-c", "user.email=builder@local", "commit", "-qm", "init", "--allow-empty")


def commit(msg):
    git("add", "-A")
    r = git("-c", "user.name=builder", "-c", "user.email=builder@local", "commit", "-qm", msg)
    return r.returncode == 0


def build_milestone(st, m):
    max_fix = int(env("BUILD_MAX_FIX", "3"))
    plan_text = read_file("PLAN.md", 8000) or ""
    existing = [f for f in project_tree() if not f.endswith((".pyc",)) and f != "PLAN.md"]
    related = list(dict.fromkeys(m["files"] + [f for f in existing if f in m["files"]]))
    ctx_rels = related + [f for f in existing if f not in related][:6]
    log("== milestone %d: %s" % (m["id"], m["title"]))
    prompt = ("PLAN:\n%s\n\nCURRENT MILESTONE %d: %s\nGoal: %s\nFiles to create or change: %s\nVerified by: %s\n\n"
              "Existing project files: %s\n\nRelevant current file contents:\n%s\n\n"
              "Implement this milestone now. Include tests for it (or update the existing tests) so the project "
              "test command `%s` exercises the new work and passes.") % (
        plan_text, m["id"], m["title"], m["goal"], ", ".join(m["files"]), m.get("tests", ""),
        ", ".join(existing) or "(none)", context_files(ctx_rels), st["test_cmd"])
    text = call_model("write", [{"role": "system", "content": SYS_WRITER}, {"role": "user", "content": prompt}], 12000)
    files = parse_files(text)
    if not files:
        log("writer returned no files")
        return False
    write_files(files)
    touched = set(files)
    # test / fix loop
    for attempt in range(max_fix + 1):
        m["attempts"] += 1
        save_state(st)
        code, out = sh(st["test_cmd"])
        if code == 0:
            log("tests pass")
            break
        out = out[-MAX_ERR_CHARS:]
        log("tests failed (attempt %d/%d)" % (attempt + 1, max_fix + 1))
        if attempt == max_fix:
            log("giving up on milestone %d; last output kept in .builder/last_failure.txt" % m["id"])
            with open(os.path.join(STATE_DIR, "last_failure.txt"), "w") as f:
                f.write(out)
            return False
        fp = ("The tests failed.\nTest command: %s\n\nOutput (tail):\n%s\n\nMilestone: %s - %s\n\n"
              "Files involved:\n%s\n\nFix the code so the tests pass. Change only what is needed, output the FULL "
              "content of every file you change, in the required format.") % (
            st["test_cmd"], out, m["title"], m["goal"], context_files(sorted(touched) + [f for f in project_tree() if f.startswith("test")][:3]))
        t2 = call_model("fix", [{"role": "system", "content": SYS_WRITER}, {"role": "user", "content": fp}], 12000)
        fixed = parse_files(t2)
        if fixed:
            write_files(fixed)
            touched |= set(fixed)
    # review the diff for this milestone
    git("add", "-A")
    diff = git("diff", "--cached").stdout[:14000]
    if diff.strip():
        rp = ("Review this diff for SECURITY problems (secrets, injection, unsafe file/shell use, missing input "
              "validation) and real BUGS. Ignore style. Reply with ONLY JSON: "
              '{"issues":[{"severity":"high|medium|low","file":"...","problem":"...","fix":"..."}]}'
              " (empty list if none).\n\nDIFF:\n%s") % diff
        issues = []
        for tryno in range(2):
            ask = rp if tryno == 0 else rp + "\n\nIMPORTANT: reply with ONLY the JSON object, no other text."
            try:
                rv = extract_json(call_model("review", [{"role": "system", "content": "You are a strict code reviewer."},
                                                        {"role": "user", "content": ask}], 3000), "issues")
                issues = rv.get("issues", []) or []
                break
            except Exception as e:
                log("review unusable (%s)%s" % (str(e)[:80], "; retrying" if tryno == 0 else "; skipped"))
        serious = [i for i in issues if str(i.get("severity", "")).lower() in ("high", "medium")]
        with open("REVIEW.md", "a") as f:
            f.write("\n## Milestone %d: %s\n" % (m["id"], m["title"]))
            f.write("\n".join("- [%s] %s: %s" % (i.get("severity"), i.get("file"), i.get("problem")) for i in issues) or "- no issues")
            f.write("\n")
        if serious:
            log("reviewer found %d serious issue(s); asking for a fix" % len(serious))
            fp = ("A reviewer found these issues in the code you wrote:\n%s\n\nFiles:\n%s\n\nFix them without "
                  "breaking the tests. Output the FULL content of every file you change.") % (
                json.dumps(serious, indent=1), context_files(sorted(touched)))
            fixed = parse_files(call_model("fix", [{"role": "system", "content": SYS_WRITER},
                                                   {"role": "user", "content": fp}], 12000))
            if fixed:
                backup = {rel: read_file(rel, 10**7) for rel in fixed}
                write_files(fixed)
                code, out = sh(st["test_cmd"])
                if code != 0:
                    log("review fix broke tests; reverting it")
                    write_files({k: v for k, v in backup.items() if v is not None})
    with open("PROGRESS.md", "a") as f:
        f.write("- milestone %d done: %s\n" % (m["id"], m["title"]))
    commit("milestone %d: %s" % (m["id"], m["title"]))
    return True


def cmd_run(push):
    st = load_state()
    ensure_git()
    for m in st["milestones"]:
        if m["status"] == "done":
            continue
        if build_milestone(st, m):
            m["status"] = "done"
            save_state(st)
        else:
            m["status"] = "failed"
            save_state(st)
            log("Stopped at milestone %d. Fix PLAN.md or the code by hand, set status back to 'pending' in "
                ".builder/state.json, and run again." % m["id"])
            return 1
    log("All milestones done.")
    if push:
        r = git("push", "-u", "origin", "HEAD")
        log("git push: %s" % ((r.stdout + r.stderr).strip()[:300] or "ok"))
    else:
        log("Review with `git log` and `git diff init`, then push yourself or rerun with --push.")
    return 0


def main(argv):
    if len(argv) < 2 or argv[1] not in ("plan", "run", "status"):
        print(__doc__)
        return 2
    if argv[1] == "plan":
        if len(argv) < 3:
            sys.exit('Usage: build.py plan "describe the project"')
        cmd_plan(" ".join(argv[2:]))
    elif argv[1] == "status":
        cmd_status()
    else:
        return cmd_run("--push" in argv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))