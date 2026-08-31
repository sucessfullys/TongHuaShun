======================================================================
  ERA  ·  Task Init complete
======================================================================

  Project   : {{PROJECT_NAME}}
  Workspace : {{WORKSPACE_PATH}}

{{SUMMARY}}

  Warnings:
{{WARNINGS}}

----------------------------------------------------------------------

Init the task done. Exit this session, then in your shell run:

    cd {{WORKSPACE_PATH}}
    claude --plugin-dir {{PLUGIN_DIR}} --dangerously-skip-permissions

  In the new session, type one literal message and press Enter:

    /era:start

Before proceeding, review `spec.md` and `config.yaml` in the workspace and
correct any mis-detected value (image roles, pairing rule, GPU reservations).
Stage 0 is complete; Stages 1+ run from `/era:start`.
