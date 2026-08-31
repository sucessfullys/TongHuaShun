# ERA Stage 8 — Human Feedback (launch the review web app, then block on Continue)

You are running ERA Stage 8 for one project workspace. Goal: start the
human-feedback web app in the background, give the operator a ready-to-paste
SSH-tunnel command and URL, **block in this same ralph-loop iteration on a
confirmation prompt**, and — once the operator selects **Continue** and the
deterministic `feedback-status` check confirms feedback is finalized — stop
the server and return so the loop advances directly into Stage 9.

Stage 8 is the **one** explicit exception to the autonomy iron rule:
`AskUserQuestion` is allowed for the wait prompt in Step 8 (the operator
hand-off is the whole point of this stage). It is **not** allowed anywhere
else in this skill — never use it to disambiguate workspace state or to
escape from a transient error.

The workspace path was passed as the skill argument (`$ARGUMENTS`). The
**ERA repo root** is the parent of the directory holding the `era/` package;
its venv Python is `<repo>/.venv/bin/python3`. Every `era.cli` call reads one
JSON object from stdin (a heredoc) and prints one out.

## Step 1 — Read state

Resolve the active iteration via `<workspace>/current` → `<iter>`. Confirm
`<iter>/experiments/results/summary.json` exists and that Stage 7 wrote
`<iter>/comparison/comparison.json`. If either is missing, append a note to
`<workspace>/logs/iterations/`, set `run_state: blocked`, and stop — Stage 8
cannot collect feedback on results that are not there.

## Step 2 — Normalize the review model for the web app

Stage 2–6 artifacts are partly agent-authored — `scores.jsonl` shapes vary by
task and evaluator. Before the web app launches, normalize them into one
artifact it can always render correctly, for *this* iteration of *any* task.

First the deterministic pass:

```bash
<repo>/.venv/bin/python3 -m era.cli build-review-model <<JSON
{"workspace_path": "<workspace>"}
JSON
```

It writes `<iter>/human/review_model.json` and returns `warning_count` +
`warnings`.

Then dispatch the **`era-review-adapter`** sub-agent **once** to repair and
enrich that artifact (factor labels, plain-language descriptions, and anything
listed in `warnings`). Pass it the workspace path and the iteration directory;
it edits only `<iter>/human/review_model.json` and ends with `VERDICT: OK`.

This step **never blocks Stage 8**: if `build-review-model` returns an `error`
or the sub-agent is unavailable, append a note to `<workspace>/logs/iterations/`
and continue to Step 3 anyway — the web app falls back to reading the raw
artifacts directly, so the review still works.

## Step 3 — Ensure the GPU watchdog is alive, then launch the web app

**3a. Make sure the GPU watchdog is protecting idle cards.** Stage 8 can wait
on the operator for a long time, during which the experiment GPUs are idle —
re-arm `NoGPUAlarmNew.py` so other users don't grab them:

```bash
<repo>/.venv/bin/python3 -m era.cli ensure-watchdog <<JSON
{"workspace_path": "<workspace>"}
JSON
```

This is the autonomous equivalent of `cd
/mnt/image-edit/datasets/xywang/code/GPU_OCU/ && bash start.sh`. It checks
actual process liveness (`pgrep`) and starts the watchdog **only** when none
is running — so it never spawns a duplicate. The reply's `action` is one of
`already_alive` / `started` / `start_skipped_no_dir` / `start_failed` /
`probe_failed`. This step **never blocks Stage 8**: on any action other than
`already_alive` / `started`, append a one-line note to
`<workspace>/logs/iterations/` and continue to 3b regardless — the operator
hand-off must not be gated on the watchdog.

**3b. Launch the feedback web app (detached, background).**

```bash
<repo>/.venv/bin/python3 -m era.cli serve-feedback <<JSON
{"workspace_path": "<workspace>"}
JSON
```

`serve-feedback` starts the web app as a **detached background process** that
keeps running after this loop iteration ends. It returns the `host`, `port`,
`url`, `pid`, and `logfile`. If it returns an `error`, retry up to **3 times**
with a short backoff (a port race, a slow first import). If it still fails,
append the error + the `logfile` path to `<workspace>/logs/iterations/`, set
`run_state: blocked`, and stop.

## Step 4 — Verify the server is up

```bash
<repo>/.venv/bin/python3 -m era.cli feedback-status <<JSON
{"workspace_path": "<workspace>"}
JSON
```

Confirm `server.running` and `server.responsive` are both `true`. Retry up to
3 times with a short backoff if not. If the server never becomes responsive,
record it, set `run_state: blocked`, and stop.

## Step 5 — Build the operator instructions

Resolve the box's hostname and the current user with Bash (`hostname` and
`whoami`). Using the `port` from Step 2, build the operator handoff block:

```
==================================================================
  ERA Stage 8 — human review required
==================================================================
  The evaluation results for <project> / <iter> are ready for your review.

  1. From your local machine, open an SSH tunnel to this box:

       ssh -N -L <port>:127.0.0.1:<port> <user>@<hostname>

     (add  -L <port>:127.0.0.1:<port>  to however you normally SSH in.)

  2. Open the review app in your local browser:

       http://localhost:<port>/

  3. Review each sample. Flag any Family-A / hybrid judgement that is wrong
     and any Family-B ranking that is wrong; everything you do NOT flag is
     recorded as correct. Add general feedback, then click "Finalize".

  4. Back in this terminal, a confirmation prompt will appear after this
     message — select **Continue** once you have clicked Finalize in the
     web app. (If you closed this terminal, run  /era:resume  to re-enter
     Stage 8 — it will detect the finalized feedback and advance.)

  The pipeline is now paused (run_state: awaiting_human).
==================================================================
```

## Step 6 — Persist the instructions and set the wait state

Append the full instruction block (with the real port, hostname, and user
filled in) to `<workspace>/logs/iterations/`, and **also print it as your
final message before the wait prompt** so the operator sees it immediately.

Then set the wait state — pass **only** `run_state`, never `stage_index`
(`/era:status` shows this state while you are blocked in Step 8):

```bash
<repo>/.venv/bin/python3 -m era.cli update-status <<JSON
{"workspace_path": "<workspace>", "run_state": "awaiting_human"}
JSON
```

The detached web server keeps running across this skill's lifetime; the
wait happens in Step 8 below.

## Step 7 — Pre-check the finalize state (re-entry shortcut)

Before blocking on the operator, check whether feedback is **already
finalized** — this is the path that fires when the operator closed the
terminal mid-wait and ran `/era:resume`, so the same Stage 8 skill is being
re-entered:

```bash
<repo>/.venv/bin/python3 -m era.cli feedback-status <<JSON
{"workspace_path": "<workspace>"}
JSON
```

If `feedback.finalized` is **`true`**, the operator is done — skip Step 8
entirely and proceed to Step 9. If it is `false`, continue to Step 8.

## Step 8 — Block on the operator confirmation prompt (the in-loop wait)

Use the `AskUserQuestion` tool. **This is the one place in Stage 8 — and the
one place in the ralph loop — where the agent may ask the operator
anything.** Pose the question once and react to the answer; loop the prompt
as needed until the operator selects Continue with feedback finalized, or
Cancels.

- **Question:** "Have you finalized the feedback in the web app?"
- **Header:** "Stage 8 wait"
- **Options (3, single-select):**
  1. **Continue to Stage 9 (Recommended)** — description: "I have clicked
     Finalize in the web app; advance the pipeline."
  2. **Still working — wait a bit** — description: "Re-prompt me; I am still
     reviewing."
  3. **Cancel this run** — description: "Stop the pipeline cleanly; I will
     resume later or restart."

AskUserQuestion automatically offers an **Other** free-text field; treat any
non-empty Other answer the same as Continue.

Branch on the answer:

- **Continue (or any Other free-text answer):** call `feedback-status` again.
  - If `feedback.finalized` is `true`, exit this step and go to Step 9.
  - If it is `false`, the operator selected Continue too early. Re-ask the
    AskUserQuestion exactly as above, but prepend the question text with the
    warning `"_Feedback is not finalized yet — click **Finalize** in the
    web app first, then try again._"`.
- **Still working — wait a bit:** re-ask the AskUserQuestion unchanged. (The
  operator returns to the browser; the web app stays up.)
- **Cancel this run:** stop the server, mark the run stopped, and return —
  the ralph loop will see `run_state: stopped` and exit cleanly:

  ```bash
  <repo>/.venv/bin/python3 -m era.cli stop-feedback <<JSON
  {"workspace_path": "<workspace>"}
  JSON
  ```

  ```bash
  <repo>/.venv/bin/python3 -m era.cli update-status <<JSON
  {"workspace_path": "<workspace>", "run_state": "stopped"}
  JSON
  ```

  Append a one-line Stage-8 note to `<workspace>/logs/iterations/` recording
  that the operator cancelled. Return.

Defensive cap: if the prompt has looped more than ~10 times without the
operator either confirming Continue with finalized feedback or selecting
Cancel, stop the server, set `run_state: awaiting_human`, log the
situation, and return — the ralph loop will exit and `/era:resume` becomes
the recovery path.

## Step 9 — Clean exit (feedback finalized; advance into Stage 9)

The operator confirmed Continue and `feedback-status` returned
`feedback.finalized: true`. Tear down the wait:

```bash
<repo>/.venv/bin/python3 -m era.cli stop-feedback <<JSON
{"workspace_path": "<workspace>"}
JSON
```

```bash
<repo>/.venv/bin/python3 -m era.cli update-status <<JSON
{"workspace_path": "<workspace>", "run_state": "running"}
JSON
```

Append a one-line Stage-8 note to `<workspace>/logs/iterations/` recording
that the operator finalized and continued. Print a brief one-line confirmation
to the terminal (e.g. `"Stage 8 done — feedback finalized, server stopped,
advancing to Stage 9."`) and return. The ralph loop's Step 8 will see
`run_state: running` and advance `stage_index` to 8; the next ralph pass
runs Stage 9.
