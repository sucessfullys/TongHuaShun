import React, { useCallback, useEffect, useMemo, useState } from "react";
import * as api from "./api.js";
import SampleSidebar from "./SampleSidebar.jsx";
import SampleView from "./SampleView.jsx";

export default function App() {
  const [overview, setOverview] = useState(null);    // /api/samples response
  const [health, setHealth] = useState(null);        // /api/health response
  const [sampleIdx, setSampleIdx] = useState(0);
  const [current, setCurrent] = useState(null);      // /api/sample/{k} response
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [savedTick, setSavedTick] = useState(0);     // bump after each save

  // Initial load: samples list + health (for the warnings banner).
  useEffect(() => {
    Promise.all([api.getOverview(), api.getHealth()])
      .then(([ov, h]) => {
        setOverview(ov);
        setHealth(h);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // Load the current sample whenever the index changes.
  useEffect(() => {
    if (!overview || !overview.samples.length) return undefined;
    const sk = overview.samples[sampleIdx]?.sample_key;
    if (!sk) return undefined;
    setLoading(true);
    let cancelled = false;
    api
      .getSample(sk)
      .then((s) => {
        if (!cancelled) {
          setCurrent(s);
          setLoading(false);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(String(e));
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [overview, sampleIdx]);

  const gotoSample = useCallback(
    (idx) => {
      if (!overview) return;
      const clamped = Math.max(0, Math.min(overview.samples.length - 1, idx));
      setSampleIdx(clamped);
    },
    [overview],
  );

  /** Save one (sample, method) annotation; on success refresh the overview
   * row + current.per_method in place so the counter + sidebar dot update. */
  const saveAnnotation = useCallback(
    async (sampleKey, methodId, text) => {
      const rec = await api.putAnnotation(sampleKey, methodId, text);
      // Mutate current.per_method to match server-side state.
      setCurrent((c) => {
        if (!c || c.sample_key !== sampleKey) return c;
        const per_method = { ...(c.per_method || {}) };
        if ((text || "").trim()) per_method[methodId] = text;
        else delete per_method[methodId];
        return { ...c, per_method, updated_at: rec.updated_at };
      });
      // Update the sidebar row's annotated flag.
      setOverview((ov) => {
        if (!ov) return ov;
        const samples = ov.samples.map((row) => {
          if (row.sample_key !== sampleKey) return row;
          // Recompute annotated state from current per_method.
          const annotated =
            ((text || "").trim() ? 1 : 0) +
            Object.entries(current?.per_method || {}).filter(
              ([m, v]) => m !== methodId && (v || "").trim(),
            ).length > 0;
          return { ...row, annotated };
        });
        const annotated_count = samples.filter((s) => s.annotated).length;
        return { ...ov, samples, annotated_count };
      });
      setSavedTick((t) => t + 1);
      return rec;
    },
    [current],
  );

  // Global keyboard shortcuts: ←/→ navigate samples (when not in an input);
  // Ctrl/Cmd+S triggers per-cell save via a custom event SampleView listens to.
  useEffect(() => {
    const onKey = (e) => {
      const inField =
        e.target.matches && e.target.matches("input, textarea");
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        window.dispatchEvent(new CustomEvent("era-annotate:save-all"));
        return;
      }
      if (inField) return;
      if (e.key === "ArrowLeft") gotoSample(sampleIdx - 1);
      else if (e.key === "ArrowRight") gotoSample(sampleIdx + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [gotoSample, sampleIdx]);

  const annotatedCount = overview?.annotated_count ?? 0;
  const totalCount = overview?.samples?.length ?? 0;
  const progressPct = totalCount
    ? Math.round((annotatedCount / totalCount) * 100)
    : 0;

  const datasetLabel = useMemo(() => {
    if (!overview?.dataset_root) return "";
    const parts = overview.dataset_root.split("/");
    return parts.slice(-2).join("/");
  }, [overview]);

  if (error) {
    return (
      <Shell datasetLabel={datasetLabel}>
        <div className="center-msg">
          <p>Could not load the annotation dataset.</p>
          <p style={{ fontFamily: "var(--mono)", fontSize: "0.8rem" }}>
            {error}
          </p>
        </div>
      </Shell>
    );
  }

  if (!overview) {
    return (
      <Shell datasetLabel={datasetLabel}>
        <div className="center-msg">
          <div className="spinner" />
          <p>Probing dataset…</p>
        </div>
      </Shell>
    );
  }

  if (!overview.samples.length) {
    return (
      <Shell datasetLabel={datasetLabel}>
        <div className="center-msg">
          <p>No samples found in this dataset.</p>
          <p style={{ fontFamily: "var(--mono)", fontSize: "0.8rem" }}>
            {overview.dataset_root}
          </p>
        </div>
      </Shell>
    );
  }

  return (
    <Shell
      datasetLabel={datasetLabel}
      annotatedCount={annotatedCount}
      totalCount={totalCount}
      progressPct={progressPct}
      sampleIdx={sampleIdx}
      gotoSample={gotoSample}
      warnings={health?.warnings || []}
    >
      <div className="annotate-body">
        <SampleSidebar
          samples={overview.samples}
          sampleIdx={sampleIdx}
          gotoSample={gotoSample}
        />
        <div className="annotate-main">
          {loading || !current ? (
            <div className="center-msg">
              <div className="spinner" />
              <p>Loading sample…</p>
            </div>
          ) : (
            <SampleView
              current={current}
              overview={overview}
              saveAnnotation={saveAnnotation}
              savedTick={savedTick}
            />
          )}
        </div>
      </div>
    </Shell>
  );
}

function Shell({
  datasetLabel,
  annotatedCount = 0,
  totalCount = 0,
  progressPct = 0,
  sampleIdx,
  gotoSample,
  warnings = [],
  children,
}) {
  const [warningsOpen, setWarningsOpen] = useState(true);
  const hasNav =
    typeof sampleIdx === "number" && typeof gotoSample === "function";
  return (
    <div className="app">
      <header className="topbar">
        <div className="wordmark">
          <span className="era">ERA</span>
          <span className="sep">·</span>
          <span className="sub">Image Annotation</span>
        </div>
        {datasetLabel && <span className="chip">{datasetLabel}</span>}
        {totalCount > 0 && (
          <>
            <span className="chip">
              {annotatedCount}/{totalCount} annotated
            </span>
            {hasNav && (
              <span className="chip">
                sample {sampleIdx + 1} of {totalCount}
              </span>
            )}
          </>
        )}
        {hasNav && (
          <nav className="nav-inline">
            <button
              className="nav-btn"
              disabled={sampleIdx <= 0}
              onClick={() => gotoSample(sampleIdx - 1)}
              title="Previous sample (←)"
            >
              ←
            </button>
            <button
              className="nav-btn"
              disabled={sampleIdx >= totalCount - 1}
              onClick={() => gotoSample(sampleIdx + 1)}
              title="Next sample (→)"
            >
              →
            </button>
          </nav>
        )}
        {totalCount > 0 && (
          <div className="progress-wrap">
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <div className="progress-label">{progressPct}% complete</div>
          </div>
        )}
      </header>
      {warnings.length > 0 && (
        <div className={`warnings-bar${warningsOpen ? " open" : ""}`}>
          <button
            className="warnings-toggle"
            onClick={() => setWarningsOpen((v) => !v)}
          >
            {warningsOpen ? "▾" : "▸"} probe warnings ({warnings.length})
          </button>
          {warningsOpen && (
            <ul className="warnings-list">
              {warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
        </div>
      )}
      <main className="main">{children}</main>
    </div>
  );
}
