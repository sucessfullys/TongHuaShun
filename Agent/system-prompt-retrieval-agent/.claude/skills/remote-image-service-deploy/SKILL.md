---
name: remote-image-service-deploy
description: >
  Deploys Image-Generater-Remote source to the 3h100 cluster via dry-run then
  real rsync, validates Python version, sets up the remote venv, and verifies
  file transfer. Always uses the SSH alias 3h100; requires explicit user
  confirmation before real rsync; never claims smoke passed.
---

## Purpose

Safely push local `Image-Generater-Remote/` changes to the remote server at
`/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote/`, create or
update the remote Python venv, and confirm file transfer. The actual remote
smoke check is run separately by `implementing-todo-item`.

The SSH alias `3h100` is defined in `~/.ssh/config`:

```text
Host 3h100
  HostName  10.217.219.2
  User      root
  Port      2891
  IdentityFile ~/.ssh/interactive-ggl0wf168o2g_rsa.key
```

Always use `3h100` — never `root@10.217.219.2` directly.

## Steps

### 0. Vendor-sync byte-equality hard gate (S00.02c)

Before any remote mutation (no `mkdir`, no dry-run, no rsync), run:

```bash
python /Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent/System-Prompt-Retrieval-Agent/scripts/sync_vendored_canonical_paths.py --check
```

Exit codes:

| Code | Meaning | Action |
| --- | --- | --- |
| 0 | Vendored copy is byte-identical to the remote source and provenance is current. | Continue to step 1. |
| 2 | Byte-mismatch, vendored copy missing, provenance missing, or provenance sha mismatch. | **ABORT.** Do not run `mkdir`, dry-run, or real rsync. Mark the ledger entry `[!]`, surface stderr to the user, and stop. |
| 3 | Master `Image-Generater-Remote/server/canonical_paths.py` missing. | **ABORT.** Same as exit 2. |

The remote filesystem must not be touched on any non-zero exit. Failure
is recoverable only by rerunning the script in default sync mode (which
re-derives the vendored copy + provenance) once the divergence is
explained.

### 1. Create remote directory

```bash
ssh 3h100 'mkdir -p /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote'
```

### 2. Dry-run rsync

```bash
rsync -av --dry-run \
  --exclude='.venv' \
  --exclude='config.yaml' \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='*.key' \
  --exclude='*.pem' \
  --exclude='runs/' \
  --exclude='outputs/' \
  --exclude='logs/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  /Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent/Image-Generater-Remote/ \
  3h100:/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote/
```

Print the full dry-run output.

### 3. Require explicit user confirmation

**STOP.** Show the dry-run output to the user and ask:

> "Real rsync ready. Proceed? (yes / no)"

Do not run real rsync until the user answers "yes". This pause is required
under the Iron Law destructive-action exception and is **not** overridden by
`autoAllowBashIfSandboxed` or any sandbox auto-allow setting.

### 4. Real rsync (only after user confirms)

```bash
rsync -av \
  --exclude='.venv' \
  --exclude='config.yaml' \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='*.key' \
  --exclude='*.pem' \
  --exclude='runs/' \
  --exclude='outputs/' \
  --exclude='logs/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  /Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent/Image-Generater-Remote/ \
  3h100:/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote/
```

### 5. Validate Python version

```bash
ssh 3h100 '/usr/bin/python --version'
```

Required: Python ≥ 3.10 (FLUX.2 / torch 2.8.0 stack). If incompatible:

- Mark the ledger entry `[!]`.
- Tell the user: "Remote `/usr/bin/python` is incompatible. Add a
  `Remote python path` entry to `CLAUDE.md` §Fixed Project Paths and retry."
- **Do not create the venv.** Stop here.

### 6. Remote venv (create or update)

If `.venv` does not yet exist on the remote, or if `requirements.txt` changed:

```bash
ssh 3h100 '/usr/bin/python -m venv \
  /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote/.venv'

ssh 3h100 '/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote/.venv/bin/pip \
  install -r \
  /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote/requirements.txt'
```

### 7. Verify file transfer

```bash
ssh 3h100 'ls /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote/'
```

Confirm expected files and directories are present.

### 8. Record in progress ledger

Mark **Remote deploy completed** in the progress ledger entry for the current
todo item. Do **not** mark "smoke passed" — smoke is a separate step in
`implementing-todo-item`.

## Output Contract

- Dry-run output summary (files that would transfer).
- User confirmation recorded.
- Confirmation that real rsync completed without errors.
- Python version check result.
- Venv status (created / updated / unchanged / blocked).
- `ls` output confirming remote file presence.
- Any blocker (unexpected dry-run files, SSH failure, incompatible Python, pip error).

## Hard Rules

- **Always run the step 0 vendor-sync `--check` before any remote
  mutation.** A non-zero exit aborts deploy with no remote files mutated.
- Always run `mkdir -p` before rsync.
- Always run dry-run before real rsync; never skip it.
- **Always pause for user confirmation after dry-run; never auto-proceed.**
- Always use the `3h100` alias; never use `root@10.217.219.2` or any raw IP.
- Deploy target is always
  `/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote/`. No other path.
- Validate Python version before venv creation; hard stop if incompatible.
- Never trigger model checkpoint downloads. If a missing checkpoint is detected,
  raise a clear error and tell the user to upload the model manually.
- Never commit `config.yaml` or raw API keys.
- Ledger field is "Remote deploy completed" — not "Remote deploy and smoke passed."
- Deploy is parent-only. Subagents (`remote-server-worker`) may prepare code
  under `Image-Generater-Remote/` locally, but only the parent can run
  dry-run review, real rsync, remote venv mutations, and deploy verification.
