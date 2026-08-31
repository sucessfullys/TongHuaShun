---
name: webapp-testing
description: >
  Validate local web applications with Playwright-driven browser checks,
  screenshots, DOM inspection, console-log capture, and server lifecycle
  helpers. Use for WebUI implementation, UI regression checks, and debugging
  frontend behavior.
source: https://github.com/ComposioHQ/awesome-claude-skills/blob/master/webapp-testing/SKILL.md?plain=1
---

# Web Application Testing

Use this skill when building or validating the V0.3 WebUI.

## Workflow

1. If the app needs a server, start it through a helper that manages lifecycle
   and port readiness. Prefer a black-box helper script over hand-managed
   background processes.
2. If a helper script exists, run it with `--help` before using it.
3. For dynamic pages, navigate with Playwright and wait for `networkidle`
   before inspecting DOM state.
4. Capture a screenshot and browser console output before making assertions.
5. Use stable selectors: roles, visible text, data attributes, or specific CSS
   selectors. Avoid brittle positional selectors when a semantic selector is
   available.
6. Always close the browser at the end of the script.

## V0.3 Verification Expectations

- Verify the main prompt-pair table loads from the API.
- Verify table filtering, sorting, and prompt-pair selection.
- Verify the detail page displays samples in `dress`, `lower`, `upper` order.
- Verify the comparison page shows both selected prompt pairs side by side.
- Verify thumbnail images load, original-image links open, and missing images
  render a clear placeholder.
- Verify score chips/bars match the API values and do not overflow at mobile
  or desktop widths.

## Notes

The original upstream skill recommends Python Playwright scripts and lifecycle
helpers for local webapp checks. Keep those checks lightweight and targeted to
the UI behavior being changed.
