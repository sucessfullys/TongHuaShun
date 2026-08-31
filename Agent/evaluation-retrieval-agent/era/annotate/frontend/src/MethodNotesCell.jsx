import React, { useEffect, useRef, useState } from "react";

/** Controlled textarea for one (sample, method) annotation.
 *  Mirrors Stage 8's JudgeCell pattern:
 *   - controlled state, dirty when value differs from last saved,
 *   - save on blur, save button, AND on Ctrl+S (App emits a custom event),
 *   - per-cell status line: "saved 14:02:35" / "saving…" / "unsaved" /
 *     "error: <msg>". */
export default function MethodNotesCell({
  sampleKey, methodId, initialText, disabled, saveAnnotation, savedTick,
}) {
  const [value, setValue] = useState(initialText || "");
  const [saved, setSaved] = useState(initialText || "");
  const [status, setStatus] = useState({ kind: "idle", text: "—" });
  const taRef = useRef(null);

  // Reset state when the slot's identity or initialText changes
  // (operator navigated to a different sample or method).
  useEffect(() => {
    setValue(initialText || "");
    setSaved(initialText || "");
    setStatus(
      initialText
        ? { kind: "ok", text: `saved` }
        : { kind: "idle", text: "—" },
    );
  }, [sampleKey, methodId, initialText]);

  const dirty = value !== saved;

  async function doSave() {
    if (!dirty) return;
    setStatus({ kind: "saving", text: "saving…" });
    try {
      await saveAnnotation(sampleKey, methodId, value);
      setSaved(value);
      setStatus({
        kind: "ok",
        text: `saved ${new Date().toLocaleTimeString()}`,
      });
    } catch (err) {
      setStatus({ kind: "err", text: String(err.message || err) });
    }
  }

  // Listen for the global Ctrl+S event from App.
  useEffect(() => {
    const onSaveAll = () => { if (!disabled) doSave(); };
    window.addEventListener("era-annotate:save-all", onSaveAll);
    return () => window.removeEventListener("era-annotate:save-all", onSaveAll);
    // Re-attach when value/saved change so the closure sees fresh data.
  }, [value, saved, disabled, sampleKey, methodId]);

  // Bump status text when savedTick changes (so the App-driven save-all
  // visually refreshes our status line even if it was already "ok").
  useEffect(() => {
    if (savedTick > 0 && !dirty && status.kind === "ok") {
      // no-op visual refresh — status already correct
    }
  }, [savedTick, dirty, status.kind]);

  return (
    <div className={"notes-cell" + (disabled ? " disabled" : "")}>
      <div className="notes-cap">
        <span className="method-cap-label">notes</span>
        <span className="method-cap-id" title={methodId}>
          {methodId}
        </span>
      </div>
      <textarea
        ref={taRef}
        className="notes-textarea"
        placeholder={
          disabled
            ? "(this sample is not present in this method)"
            : `Notes about ${methodId}'s result…`
        }
        value={value}
        onChange={(e) => {
          setValue(e.target.value);
          if (status.kind === "ok") setStatus({ kind: "pending", text: "unsaved" });
        }}
        onBlur={doSave}
        disabled={disabled}
      />
      <div className="notes-row">
        <button
          type="button"
          className="notes-save-btn"
          disabled={disabled || !dirty}
          onClick={doSave}
        >
          Save
        </button>
        <span className={`notes-status status-${status.kind}`}>
          {status.text}
        </span>
      </div>
    </div>
  );
}
