#!/usr/bin/env python3
"""tg_builder.py - control build.py from Telegram with / commands.

Run:   python3 tg_builder.py
Needs: BUILDER_BOT_TOKEN (a NEW bot from @BotFather - do not reuse your other bot's token)
       TELEGRAM_ADMIN_USER_ID (your numeric Telegram id; nobody else is answered)
       plus the provider keys build.py needs (GEMINI_API_KEYS, GROQ_API_KEY, ...)
Optional: BUILDER_HOME (default ~/builds)  BUILD_SCRIPT (default build.py next to this file)

Standard library only. One build job at a time, so free quotas are not split.
"""
import json, os, re, signal, subprocess, sys, threading, time, urllib.request, urllib.error

TOKEN = os.environ.get("BUILDER_BOT_TOKEN", "").strip()
ADMIN = os.environ.get("TELEGRAM_ADMIN_USER_ID", "").strip()
HOME = os.path.expanduser(os.environ.get("BUILDER_HOME", "~/builds"))
SCRIPT = os.environ.get("BUILD_SCRIPT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "build.py")
API_BASE = os.environ.get("BUILDER_API_BASE", "https://api.telegram.org")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
REMOTE_RE = re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(\.git)?$")
CURRENT_FILE = os.path.join(HOME, ".current")

LOCK = threading.Lock()
JOB = None          # {"name","proc","kind","chat"}

COMMANDS = [
    ("new", "new <name> <idea> - make a plan for a new project"),
    ("plan", "show the current project's plan"),
    ("build", "build all pending milestones"),
    ("status", "milestone status + whether a job is running"),
    ("log", "last lines of the build log"),
    ("review", "what the reviewer found"),
    ("stop", "stop the running job (progress is saved)"),
    ("projects", "list projects"),
    ("use", "use <name> - switch project"),
    ("remote", "remote <https github url> - set where to push"),
    ("push", "push the project to GitHub"),
    ("replan", "redo the plan (only before any milestone is built)"),
    ("help", "show commands"),
]


# ----------------------------------------------------------------- telegram
def api(method, **params):
    req = urllib.request.Request("%s/bot%s/%s" % (API_BASE, TOKEN, method),
                                 data=json.dumps(params).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=70) as r:
        return json.loads(r.read().decode()).get("result")


def split_text(text, limit=3800):
    text = text.strip() or "(empty)"
    out = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        cut = cut if cut > limit // 2 else limit
        out.append(text[:cut])
        text = text[cut:].lstrip("\n")
    out.append(text)
    return out


def send(chat, text):
    for part in split_text(text):
        try:
            api("sendMessage", chat_id=chat, text=part)
        except Exception as e:  # never crash the bot over a send
            print("send failed:", str(e)[:150], flush=True)


# ----------------------------------------------------------------- helpers
def get_current():
    try:
        with open(CURRENT_FILE) as f:
            n = f.read().strip()
        return n if NAME_RE.match(n) and os.path.isdir(os.path.join(HOME, n)) else None
    except OSError:
        return None


def set_current(name):
    os.makedirs(HOME, exist_ok=True)
    with open(CURRENT_FILE, "w") as f:
        f.write(name)


def proj_path(name):
    return os.path.join(HOME, name)


def read_tail(path, lines=20, chars=3000):
    try:
        with open(path, errors="replace") as f:
            t = "".join(f.readlines()[-lines:])
    except OSError:
        return ""
    return t[-chars:]


def read_text(path, cap=6000):
    try:
        with open(path, errors="replace") as f:
            return f.read(cap)
    except OSError:
        return ""


def state_of(name):
    try:
        with open(os.path.join(proj_path(name), ".builder", "state.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def summary(name):
    st = state_of(name)
    if not st:
        return "%s (no plan yet)" % name
    ms = st.get("milestones", [])
    done = sum(1 for m in ms if m.get("status") == "done")
    failed = sum(1 for m in ms if m.get("status") == "failed")
    return "%s: %d/%d done%s" % (name, done, len(ms), (", %d failed" % failed) if failed else "")


# ----------------------------------------------------------------- jobs
def start_job(chat, name, args, kind):
    global JOB
    with LOCK:
        if JOB is not None:
            return "Busy: a %s job is running on '%s'. Use /stop or wait." % (JOB["kind"], JOB["name"])
        path = proj_path(name)
        os.makedirs(os.path.join(path, ".builder"), exist_ok=True)
        logf = open(os.path.join(path, ".builder", "job.log"), "wb")
        try:
            p = subprocess.Popen([sys.executable, SCRIPT] + args, cwd=path, stdout=logf,
                                 stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
        finally:
            logf.close()
        JOB = {"name": name, "proc": p, "kind": kind, "chat": chat, "stopped": False}
        job = JOB
    threading.Thread(target=watch, args=(job,), daemon=True).start()
    return None


def watch(job):
    global JOB
    rc = job["proc"].wait()
    name, chat, kind = job["name"], job["chat"], job["kind"]
    with LOCK:
        if JOB is job:
            JOB = None
    path = proj_path(name)
    if job["stopped"]:
        send(chat, "Stopped '%s'. Progress is saved; /build continues where it left off." % name)
        return
    tail = read_tail(os.path.join(path, ".builder", "job.log"), 15)
    if kind == "plan":
        plan = read_text(os.path.join(path, "PLAN.md"))
        if rc == 0 and plan:
            send(chat, "Plan ready for '%s':\n\n%s\n\nSend /build to start, or /replan to redo it." % (name, plan))
        else:
            send(chat, "Planning failed (exit %s):\n%s" % (rc, tail))
    elif kind == "run":
        head = "Build finished" if rc == 0 else "Build stopped (exit %s)" % rc
        send(chat, "%s for '%s'.\n%s\n\nLast log lines:\n%s" % (head, name, summary(name), tail))
    else:
        send(chat, "%s (exit %s):\n%s" % (kind, rc, tail))


# ----------------------------------------------------------------- commands
HELP = "Commands:\n" + "\n".join("/%s" % d for _, d in COMMANDS)


def need_project(chat):
    n = get_current()
    if not n:
        send(chat, "No project selected. Use /new <name> <idea> or /use <name>.")
    return n


def c_new(chat, args):
    parts = args.split(None, 1)
    if len(parts) < 2:
        return send(chat, "Usage: /new <name> <idea>\nName: lowercase letters, digits, - or _.")
    name, idea = parts[0].lower(), parts[1].strip()
    if not NAME_RE.match(name):
        return send(chat, "Bad name. Use lowercase letters, digits, - or _ (max 40).")
    if os.path.exists(proj_path(name)):
        return send(chat, "'%s' already exists. Pick another name or /use it." % name)
    err = start_job(chat, name, ["plan", idea], "plan")
    if err:
        return send(chat, err)
    set_current(name)
    send(chat, "Planning '%s'... I'll send the plan when it's ready." % name)


def c_plan(chat, args):
    n = need_project(chat)
    if n:
        send(chat, read_text(os.path.join(proj_path(n), "PLAN.md")) or "No plan yet for '%s'." % n)


def c_build(chat, args):
    n = need_project(chat)
    if not n:
        return
    if not state_of(n):
        return send(chat, "No plan yet for '%s'. Use /new or wait for planning to finish." % n)
    err = start_job(chat, n, ["run"], "run")
    send(chat, err or "Building '%s'. It can take a while on free quotas; I'll message you when it stops. /status and /log show progress." % n)


def c_status(chat, args):
    n = need_project(chat)
    if not n:
        return
    st = state_of(n)
    lines = [summary(n)]
    if st:
        for m in st.get("milestones", []):
            lines.append("[%s] %s. %s" % (m.get("status"), m.get("id"), m.get("title")))
    if JOB:
        lines.append("\nRunning: %s on '%s'" % (JOB["kind"], JOB["name"]))
    send(chat, "\n".join(lines))


def c_log(chat, args):
    n = need_project(chat)
    if n:
        t = read_tail(os.path.join(proj_path(n), ".builder", "build.log"), 25) or \
            read_tail(os.path.join(proj_path(n), ".builder", "job.log"), 25)
        send(chat, t or "No log yet.")


def c_review(chat, args):
    n = need_project(chat)
    if n:
        t = read_text(os.path.join(proj_path(n), "REVIEW.md"))
        send(chat, t[-3500:] if t else "No review notes yet.")


def c_stop(chat, args):
    global JOB
    with LOCK:
        job = JOB
    if not job:
        return send(chat, "Nothing is running.")
    job["stopped"] = True
    try:
        os.killpg(os.getpgid(job["proc"].pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    send(chat, "Stopping...")


def c_projects(chat, args):
    try:
        names = sorted(d for d in os.listdir(HOME) if NAME_RE.match(d) and os.path.isdir(proj_path(d)))
    except OSError:
        names = []
    cur = get_current()
    send(chat, "\n".join(("* " if n == cur else "  ") + summary(n) for n in names) or "No projects yet. Use /new.")


def c_use(chat, args):
    name = args.strip().lower()
    if not NAME_RE.match(name) or not os.path.isdir(proj_path(name)):
        return send(chat, "No such project. See /projects.")
    set_current(name)
    send(chat, "Using '%s'.\n%s" % (name, summary(name)))


def c_remote(chat, args):
    n = need_project(chat)
    if not n:
        return
    url = args.strip()
    if not REMOTE_RE.match(url):
        return send(chat, "Usage: /remote https://github.com/OWNER/REPO.git (https only)")
    path = proj_path(n)
    r = subprocess.run(["git", "-C", path, "remote", "get-url", "origin"], capture_output=True, text=True)
    cmd = ["git", "-C", path, "remote", "set-url" if r.returncode == 0 else "add", "origin", url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    send(chat, "Remote set." if r.returncode == 0 else "Failed: " + (r.stderr or r.stdout)[:300] +
         "\n(Build at least one milestone first so the git repo exists.)")


def c_push(chat, args):
    n = need_project(chat)
    if not n:
        return
    err = start_job(chat, n, ["run", "--push"], "push")
    send(chat, err or "Pushing '%s' (finishes any pending milestones first)..." % n)


def c_replan(chat, args):
    n = need_project(chat)
    if not n:
        return
    st = state_of(n)
    if not st:
        return send(chat, "No plan to redo; use /new.")
    if any(m.get("status") == "done" for m in st.get("milestones", [])):
        return send(chat, "Milestones are already built; replanning would orphan them. Use /new with a new name.")
    idea = st.get("idea", "")
    if not idea:
        return send(chat, "Original idea not found; use /new with a new name.")
    for rel in (os.path.join(".builder", "state.json"), "PLAN.md"):
        try:
            os.remove(os.path.join(proj_path(n), rel))
        except OSError:
            pass
    err = start_job(chat, n, ["plan", idea], "plan")
    send(chat, err or "Re-planning '%s'..." % n)


HANDLERS = {"new": c_new, "plan": c_plan, "build": c_build, "status": c_status, "log": c_log,
            "review": c_review, "stop": c_stop, "projects": c_projects, "use": c_use, "remote": c_remote,
            "push": c_push, "replan": c_replan, "help": lambda chat, a: send(chat, HELP),
            "start": lambda chat, a: send(chat, HELP)}


def handle_update(u):
    msg = u.get("message") or {}
    text = (msg.get("text") or "").strip()
    uid = str((msg.get("from") or {}).get("id", ""))
    chat = (msg.get("chat") or {}).get("id")
    if not text or chat is None or not ADMIN or uid != ADMIN:
        return  # silently ignore everyone except the admin
    if not text.startswith("/"):
        return send(chat, "Send a / command. /help lists them.")
    head, _, args = text.partition(" ")
    cmd = head[1:].split("@")[0].lower()
    fn = HANDLERS.get(cmd)
    if not fn:
        return send(chat, "Unknown command. /help")
    try:
        fn(chat, args.strip())
    except Exception as e:
        send(chat, "Error in /%s: %s" % (cmd, str(e)[:300]))


def main():
    if not TOKEN or not ADMIN:
        sys.exit("Set BUILDER_BOT_TOKEN and TELEGRAM_ADMIN_USER_ID first.")
    if not os.path.exists(SCRIPT):
        sys.exit("build.py not found at %s (set BUILD_SCRIPT)." % SCRIPT)
    os.makedirs(HOME, exist_ok=True)
    try:
        api("setMyCommands", commands=[{"command": c, "description": d[:250]} for c, d in COMMANDS])
    except Exception as e:
        print("setMyCommands failed:", str(e)[:150], flush=True)
    print("builder bot running; projects in %s" % HOME, flush=True)
    offset = 0
    while True:
        try:
            ups = api("getUpdates", offset=offset, timeout=50, allowed_updates=["message"]) or []
        except Exception as e:
            print("poll error:", str(e)[:150], flush=True)
            time.sleep(5)
            continue
        for u in ups:
            offset = u["update_id"] + 1
            handle_update(u)


if __name__ == "__main__":
    main()