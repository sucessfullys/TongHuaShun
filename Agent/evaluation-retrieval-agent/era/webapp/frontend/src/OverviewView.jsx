import React, { useEffect, useState } from "react";
import * as api from "./api.js";

export default function OverviewView() {
  const [ov, setOv] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.getOverview().then(setOv).catch((e) => setError(String(e)));
  }, []);

  if (error)
    return <div className="center-msg">Could not load the overview. {error}</div>;
  if (!ov)
    return (
      <div className="center-msg">
        <div className="spinner" />
      </div>
    );

  const summary = ov.summary;
  const findings = ov.final_comparison?.cross_family_findings;
  const diag = ov.diagnostics;

  return (
    <div className="ov-grid">
      <div className="panel ov-card">
        <h3>Evaluator configurations</h3>
        <p style={{ color: "var(--muted)", fontSize: "0.85rem", marginBottom: "0.7rem" }}>
          Each configuration scored every result independently. The human review
          decides which of these agrees with your judgement.
        </p>
        {summary?.configs ? (
          <table className="tbl">
            <thead>
              <tr>
                <th>Configuration</th>
                <th>Family</th>
                <th>Mean score</th>
                <th>Scored</th>
              </tr>
            </thead>
            <tbody>
              {summary.configs.map((c) => (
                <tr key={c.combination_id}>
                  <td>{c.combination_id}</td>
                  <td>
                    <span className={"fam " + c.family}>{c.family}</span>
                  </td>
                  <td className="num">
                    {c.score_mean == null ? "—" : c.score_mean.toFixed(4)}
                  </td>
                  <td className="num">{c.ok_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p style={{ color: "var(--muted)" }}>No result summary found.</p>
        )}
      </div>

      {findings && findings.length > 0 && (
        <div className="panel ov-card">
          <h3>Cross-family findings</h3>
          {findings.map((f, i) => (
            <div className="finding" key={i}>
              {f}
            </div>
          ))}
        </div>
      )}

      {diag && (
        <div className="panel ov-card">
          <h3>Family A vs Family B — disagreement diagnostics</h3>
          {"rankings_agree" in diag && (
            <div className="finding">
              The VLM-judge ranking and the metric ranking{" "}
              <strong>
                {diag.rankings_agree ? "agree" : "disagree"}
              </strong>
              .
            </div>
          )}
          {diag.beautification_realism_gap_postprocessed_method && (
            <div className="finding">
              Beautification realism gap on{" "}
              <code>
                {diag.beautification_realism_gap_postprocessed_method.method}
              </code>
              : {String(
                diag.beautification_realism_gap_postprocessed_method
                  .mean_vlm_minus_metric_realism,
              )}
            </div>
          )}
          <details style={{ marginTop: "0.6rem" }}>
            <summary
              style={{
                cursor: "pointer",
                color: "var(--muted)",
                fontSize: "0.82rem",
              }}
            >
              Raw diagnostics
            </summary>
            <pre
              style={{
                fontFamily: "var(--mono)",
                fontSize: "0.72rem",
                color: "var(--muted)",
                overflowX: "auto",
                marginTop: "0.5rem",
                lineHeight: 1.5,
              }}
            >
              {JSON.stringify(diag, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
}
