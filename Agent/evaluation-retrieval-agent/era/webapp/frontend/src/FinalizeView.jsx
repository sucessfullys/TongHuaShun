import React, { useMemo, useState } from "react";
import * as api from "./api.js";

export default function FinalizeView({ data, feedback, setGeneral, onFinalized }) {
  const [text, setText] = useState(feedback?.general_feedback || "");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const finalized = feedback?.status === "finalized";

  const stats = useMemo(() => {
    let flagTotal = 0;
    let cmpTotal = 0;
    for (const s of data.samples) {
      for (const cell of s.cells) flagTotal += cell.judges.length;
      cmpTotal += (s.family_b_rankings || []).length;
    }
    return {
      flagTotal,
      flagWrong: (feedback?.item_marks || []).length,
      cmpTotal,
      cmpWrong: (feedback?.comparison_marks || []).length,
    };
  }, [data, feedback]);

  const doFinalize = async (reopen) => {
    setBusy(true);
    const res = await api.finalize(reopen);
    setBusy(false);
    setResult(res);
    if (res.ok) {
      const fresh = await api.getReview();
      onFinalized(fresh.feedback);
    }
  };

  return (
    <div className="finalize-grid">
      <div className="summary-cards">
        <div className="stat">
          <div className="big endorse">{stats.flagTotal - stats.flagWrong}</div>
          <div className="lbl">A / hybrid endorsed</div>
        </div>
        <div className="stat">
          <div className="big flag">{stats.flagWrong}</div>
          <div className="lbl">A / hybrid flagged wrong</div>
        </div>
        <div className="stat">
          <div className="big endorse">{stats.cmpTotal - stats.cmpWrong}</div>
          <div className="lbl">Family B rankings correct</div>
        </div>
        <div className="stat">
          <div className="big flag">{stats.cmpWrong}</div>
          <div className="lbl">Family B rankings wrong</div>
        </div>
      </div>

      <div className="panel ov-card">
        <h3>General feedback</h3>
        <p style={{ color: "var(--muted)", fontSize: "0.85rem", marginBottom: "0.6rem" }}>
          Anything about the evaluation as a whole — which method family tracked
          your judgement, systematic blind spots, what a better protocol should do.
        </p>
        <textarea
          className="general-area"
          value={text}
          disabled={finalized}
          onChange={(e) => setText(e.target.value)}
          onBlur={() => setGeneral(text)}
          placeholder="Overall feedback for the next iteration…"
        />
      </div>

      <div className="panel ov-card">
        <h3>Finalize the review</h3>
        <p style={{ color: "var(--muted)", fontSize: "0.85rem", marginBottom: "0.8rem" }}>
          Finalizing writes <code>human_labels.json</code> — every Family-A /
          hybrid result you did <strong>not</strong> flag is recorded as
          endorsed/correct, and every Family-B ranking you did not mark wrong is
          recorded as correct. Then return to the terminal and run{" "}
          <code>/era:resume</code>.
        </p>

        {result && result.ok && (
          <div className="banner ok" style={{ marginBottom: "0.8rem" }}>
            Feedback finalized. {result.body?.labels?.flagging_label_count}{" "}
            flagging labels and {result.body?.labels?.comparison_label_count}{" "}
            comparison labels written to{" "}
            <code>{result.body?.human_labels_json}</code>. You can now run{" "}
            <code>/era:resume</code>.
          </div>
        )}
        {result && !result.ok && result.status === 409 && (
          <div className="banner warn" style={{ marginBottom: "0.8rem" }}>
            This feedback was already finalized. Re-finalize to overwrite the
            labels with your latest edits.
          </div>
        )}
        {result && !result.ok && result.status !== 409 && (
          <div className="banner warn" style={{ marginBottom: "0.8rem" }}>
            Finalize failed ({result.status}). {String(result.body?.detail || "")}
          </div>
        )}

        <div style={{ display: "flex", gap: "0.6rem", alignItems: "center" }}>
          {!finalized && (
            <button
              className="btn-primary"
              onClick={() => doFinalize(false)}
              disabled={busy}
            >
              {busy ? "Finalizing…" : "Finalize feedback"}
            </button>
          )}
          {(finalized || (result && result.status === 409)) && (
            <button
              className="btn-ghost"
              onClick={() => doFinalize(true)}
              disabled={busy}
            >
              {busy ? "Re-finalizing…" : "Re-finalize with latest edits"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
