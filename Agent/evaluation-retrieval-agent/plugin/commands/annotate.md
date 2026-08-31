---
description: "Launch the image-annotation web app for a multi-method dataset"
argument-hint: "<dataset_root> | <mission paragraph containing a dataset path>"
---

# /era:annotate

Launch a standalone web app for browsing a multi-method try-on / image
dataset and typing per-method free-text annotations. Annotations save into
the dataset directory at `<dataset>/annotations/<sample_key>.json` for
later pickup by Stage 7 (problem display) and Stage 9 (evolution input).

This command is **not tied to an ERA workspace** — it operates directly
against the dataset path you pass in (or describe in a mission paragraph).

## Setup

Resolve these once:

- **`REPO`** — the ERA repo root: `${CLAUDE_PLUGIN_ROOT}/..` (contains
  `era/`, `plugin/`, `workspaces/`).
- **`PY`** — `${CLAUDE_PLUGIN_ROOT}/../.venv/bin/python3`.

## Steps

This flow is **mandatory in order** — do NOT skip a step. Skipping the
probe (Step 2) is exactly how a previous run produced "no images at
all" — the server started against a wrong-shape path and the operator
got nothing in the browser. Stage 0 (`/era:init`) follows the same
pattern: read the operator's mission, probe, present findings, then
act.

### Step 1 — Extract the dataset root from the operator's mission

`$ARGUMENTS` may be:

- A bare absolute path (e.g. `/mnt/.../tryon_results`).
- A relative path.
- A **multi-line mission paragraph** describing the dataset — typically
  containing one or more absolute paths plus prose like "Each sample
  has input_cloth.png, …". This is the common case; treat it like
  `/era:init`'s Stage 0 §1–§3.

Procedure:

1. Scan `$ARGUMENTS` for absolute filesystem paths
   (`/mnt/...`, `/data/...`, `/home/...`, etc.).
2. If exactly one path is plausible as the dataset root, use it.
3. If multiple plausible paths, use **AskUserQuestion** with the
   candidates as options so the operator picks the right one.
4. If no path is found at all, use **AskUserQuestion** to ask the
   operator for it.
5. If `$ARGUMENTS` is empty, ask the operator for both the path and
   any constraints they want to add (this mirrors `/era:init`'s empty
   handling).

Set `DATASET` to the resolved absolute path. Print one line so the
operator sees what you settled on: `Dataset root: <DATASET>`.

### Step 2 — Sanity-check the path and run the probe

Verify the path exists + is readable, then probe its layout:

```bash
ls -la "<DATASET>" | head
```

```bash
"$PY" -m era.cli annotate-probe <<JSON
{"dataset_root": "<DATASET>"}
JSON
```

Branch on the response:

- **`error: no_dataset`** → the path doesn't exist or isn't a
  directory. Print the message and stop. Do **not** call
  `serve-annotate`.
- **`sample_count == 0`** → the probe found zero sample directories.
  Print: *"the probe found 0 sample directories under <DATASET> — the
  layout may differ from `<root>/<method>/<sample>/<images>`. Run
  `ls -la <DATASET>` to inspect. Not launching the server."*. Stop.
- **`first_sample_resolves` has any `false` value for any
  (method, role)** → at least one image won't load. Print the
  per-method, per-role resolution table verbatim, plus a one-line
  summary ("method X is missing role Y — the operator's output_file
  override may be needed"). Stop and ask the operator to triage —
  do not start a server that would 404 on first image fetch.
- Otherwise → continue to Step 3.

### Step 3 — Show the operator the probe summary

Print this box verbatim, substituting real values from the probe
response. Do **not** leave any `<placeholder>` text in the output:

```
==================================================================
  /era:annotate — probed dataset layout
==================================================================
  Dataset      : <DATASET>
  Methods (N)  : <method_id_1>
                 <method_id_2>
                 …
  Samples      : <sample_count> (sample_glob: <sample_glob>)
  First sample : <first_sample_key>
  Input roles  : <role_1>=<filename>, <role_2>=<filename>
  Per-method outputs:
    <method_id_1> → <output_file_1>
    <method_id_2> → <output_file_2>
    …
  Confidence   : <high | needs_confirmation>
==================================================================
```

If `warnings` is non-empty, print a bullet list under the box:

```
  Probe warnings:
    - <warning 1>
    - <warning 2>
```

This is the operator's only chance to spot a probe miss **before** the
server eats GPU/RAM and they open a browser to nothing.

### Step 4 — Confirm ambiguous items (only when confidence != "high")

When `confidence == "high"`, **skip this step** and go straight to
Step 5.

When `confidence == "needs_confirmation"`, use `AskUserQuestion` to
disambiguate the items the probe couldn't resolve cleanly:

- **Per-method output file** — for every method in `output_candidates`
  whose list has more than one entry, ask:
  *"Which file is method X's final generated output?"* with each
  candidate as an option. Record the choice as
  `output_overrides[method_X] = <chosen filename>`.
- **Missing input roles** — when the probe's `input_roles` is empty
  (it couldn't identify input files by name), ask the operator to
  name the input filenames (one AskUserQuestion per expected role,
  with a free-text "Other" path). Record as
  `input_role_overrides[role_X] = <chosen filename>`.

Skip the confirmation entirely if the probe already returned
clean choices for everything (single `output_candidates` entry per
method and non-empty `input_roles`).

### Step 5 — Start the server and print the handoff block

5a. **Check whether a server is already running** for this dataset:

```bash
"$PY" -m era.cli annotate-status <<JSON
{"dataset_root": "<DATASET>"}
JSON
```

If `server.running` and `server.responsive` are both `true`, the app
is already up — do **not** start a second server. Use the returned
`pid`/`host`/`port`/`url` as the values for the handoff block and
jump to 5d.

5b. **Start the server (background, detached):**

```bash
"$PY" -m era.cli serve-annotate <<JSON
{"dataset_root": "<DATASET>",
 "output_overrides": <output_overrides from Step 4, or {}>,
 "input_role_overrides": <input_role_overrides from Step 4, or {}>}
JSON
```

Omit the two `*_overrides` fields entirely (don't even send `{}`) when
Step 4 was skipped — keeps the JSON clean.

On `error`, print the message and stop. On success the response carries
`pid`, `host`, `port`, `url`, `pidfile`, `logfile`, `responsive`. If
`responsive` is `false`, point the operator at the logfile and stop —
the server failed to come up.

5c. **Resolve host identity** so the SSH-tunnel command is concrete:

```bash
hostname
whoami
```

Let `HOST` be the `hostname` output, `USER` be the `whoami` output,
`PORT` be the `port` returned by 5b, `PID` be its `pid`, and `URL` be
its `url`.

5d. **Print the operator handoff block verbatim** (substitute the real
values — do not leave any `<…>` placeholders). The server binds to
`127.0.0.1` only, so a tunnel is required when annotating from a
different machine.

```
==================================================================
  ERA — image annotation web app is running
==================================================================
  Dataset      : <DATASET>
  Methods (N)  : <method_id_1>, <method_id_2>, …
  Samples      : <sample_count> total, 0 annotated so far
  Server       : http://127.0.0.1:<PORT>/ (bound to localhost on <HOST>)

  1. From your local machine, open an SSH tunnel to this box.
     If you are NOT on <HOST>, run this first (leave it running
     in its own terminal — close it with Ctrl+C when done):

       ssh -N -L <PORT>:127.0.0.1:<PORT> <USER>@<HOST>

     (Or add  -L <PORT>:127.0.0.1:<PORT>  to your usual ssh command.
      If you are already on <HOST>, skip this step.)

  2. Open the annotation app in your local browser:

       http://localhost:<PORT>/

  3. For each sample, look at the input + every method's tryon_result,
     then type your notes in the per-method textarea and click Save
     (or press Ctrl+S to save all dirty textareas). Keys: ← / → to
     navigate samples. Empty notes are fine — just leave the box blank
     and move on.

  4. Annotations save here, one file per sample:

       <DATASET>/annotations/<sample_key>.json

     Stage 7 picks these up to surface known problems in the next
     review round; Stage 9 React folds them into evolution_state.

  5. When you are done, stop the server from this terminal:

       "$PY" -m era.cli stop-annotate <<JSON
       {"dataset_root": "<DATASET>"}
       JSON

     (Or just kill <PID> if you closed this terminal — the PID is
      stored in <DATASET>/.annotate_server.pid.)
==================================================================
```

Then stop — the server is up; the rest is the operator's call.

## Notes

- The server runs detached; closing this Claude session does not stop it.
- Re-running `/era:annotate <DATASET>` is safe — Step 5a detects the
  existing server and reuses it. The probe also runs again so you see
  fresh findings.
- Layout: `<root>/<method>/<sample_key>/<role_filenames>`. The probe
  auto-handles depth-N sample trees (e.g.
  `<root>/<method>/<category>/<background>/<sample_dir>/`). Hidden /
  dot-prefixed entries are ignored at every level.
- `metadata.json` (if present per sample) is read **only** when
  external tooling asks for source tracing; the annotation app never
  treats it as evaluation ground truth.
