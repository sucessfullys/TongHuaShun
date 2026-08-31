import React, { useCallback, useEffect, useMemo, useState } from "react";
import * as api from "./api.js";
import ReviewView from "./ReviewView.jsx";
import OverviewView from "./OverviewView.jsx";
import FinalizeView from "./FinalizeView.jsx";

const TABS = [
  ["review", "Review"],
  ["overview", "Overview"],
  ["finalize", "Finalize"],
];

export default function App() {
  const [data, setData] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("review");
  const [sampleIdx, setSampleIdx] = useState(0);
  const [visited, setVisited] = useState(() => new Set([0]));

  useEffect(() => {
    api
      .getReview()
      .then((d) => {
        setData(d);
        setFeedback(d.feedback);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // Keyboard navigation on the Review tab.
  useEffect(() => {
    if (tab !== "review" || !data) return undefined;
    const onKey = (e) => {
      if (e.target.matches("input, textarea")) return;
      if (e.key === "ArrowLeft") gotoSample(sampleIdx - 1);
      else if (e.key === "ArrowRight") gotoSample(sampleIdx + 1);
      else if (e.key === "s") api.saveDraft();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const gotoSample = useCallback(
    (idx) => {
      if (!data) return;
      const clamped = Math.max(0, Math.min(data.samples.length - 1, idx));
      setSampleIdx(clamped);
      setVisited((v) => new Set(v).add(clamped));
    },
    [data],
  );

  const flagItem = useCallback(async (mark) => {
    const res = await api.putItem(mark);
    setFeedback(res.feedback);
  }, []);

  const markComparison = useCallback(async (mark) => {
    const res = await api.putComparison(mark);
    setFeedback(res.feedback);
  }, []);

  const setGeneral = useCallback(async (text) => {
    const res = await api.putGeneral(text);
    setFeedback(res.feedback);
  }, []);

  const lookups = useMemo(() => {
    const items = feedback?.item_marks || [];
    const cmps = feedback?.comparison_marks || [];
    return {
      item: (s, m, c) =>
        items.find(
          (x) =>
            x.sample_key === s &&
            x.method_id === m &&
            x.combination_id === c,
        ) || null,
      comparison: (s, c) =>
        cmps.find((x) => x.sample_key === s && x.combination_id === c) || null,
    };
  }, [feedback]);

  if (error)
    return (
      <Shell tab={tab} setTab={setTab} data={null}>
        <div className="center-msg">
          <p>Could not load the review data.</p>
          <p style={{ fontFamily: "var(--mono)", fontSize: "0.8rem" }}>
            {error}
          </p>
        </div>
      </Shell>
    );

  if (!data)
    return (
      <Shell tab={tab} setTab={setTab} data={null}>
        <div className="center-msg">
          <div className="spinner" />
          <p>Loading evaluation results…</p>
        </div>
      </Shell>
    );

  return (
    <Shell tab={tab} setTab={setTab} data={data}>
      {tab === "review" && (
        <ReviewView
          data={data}
          sampleIdx={sampleIdx}
          gotoSample={gotoSample}
          visited={visited}
          lookups={lookups}
          flagItem={flagItem}
          markComparison={markComparison}
          finalized={feedback?.status === "finalized"}
        />
      )}
      {tab === "overview" && <OverviewView />}
      {tab === "finalize" && (
        <FinalizeView
          data={data}
          feedback={feedback}
          setGeneral={setGeneral}
          onFinalized={setFeedback}
        />
      )}
    </Shell>
  );
}

function Shell({ tab, setTab, data, children }) {
  return (
    <div className="app">
      <header className="topbar">
        <div className="wordmark">
          <span className="era">ERA</span>
          <span className="sep">·</span>
          <span className="sub">Human Feedback</span>
        </div>
        {data && (
          <>
            <span className="chip">
              {data.iteration_dir} · {data.mode}
            </span>
            <span className="chip">{data.task_family}</span>
          </>
        )}
        <nav className="tabs">
          {TABS.map(([id, label]) => (
            <button
              key={id}
              className={"tab" + (tab === id ? " active" : "")}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>
      <main className="main">{children}</main>
    </div>
  );
}
