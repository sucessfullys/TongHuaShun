import React, { useEffect, useMemo, useRef, useState } from "react";

/** A scrollable list of every sample_key in the dataset.
 *  Each row shows a (truncated middle) sample_key + an "annotated" coral
 *  dot when any per-method slot has text. Click jumps to that sample;
 *  the current sample is highlighted. */
export default function SampleSidebar({ samples, sampleIdx, gotoSample }) {
  const [filter, setFilter] = useState("");
  const listRef = useRef(null);
  const activeRowRef = useRef(null);

  // Scroll the active sample into view when sampleIdx changes externally
  // (e.g. ← / → keyboard navigation from elsewhere).
  useEffect(() => {
    if (activeRowRef.current && listRef.current) {
      activeRowRef.current.scrollIntoView({
        block: "nearest", behavior: "smooth",
      });
    }
  }, [sampleIdx]);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return samples.map((s, i) => ({ ...s, originalIdx: i }));
    return samples
      .map((s, i) => ({ ...s, originalIdx: i }))
      .filter((s) => s.sample_key.toLowerCase().includes(q));
  }, [samples, filter]);

  return (
    <aside className="sample-sidebar">
      <div className="sidebar-head">
        <div className="section-label">Samples</div>
        <input
          className="sidebar-filter"
          type="text"
          placeholder="filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
      </div>
      <div className="sidebar-list" ref={listRef}>
        {filtered.length === 0 ? (
          <div className="sidebar-empty">No samples match.</div>
        ) : (
          filtered.map((s) => {
            const active = s.originalIdx === sampleIdx;
            return (
              <button
                key={s.sample_key}
                ref={active ? activeRowRef : null}
                className={"sidebar-row" + (active ? " active" : "")}
                onClick={() => gotoSample(s.originalIdx)}
                title={s.sample_key}
              >
                <span
                  className={"dot" + (s.annotated ? " filled" : "")}
                  aria-label={s.annotated ? "annotated" : "not annotated"}
                />
                <span className="sample-key">
                  {truncateMiddle(s.sample_key, 36)}
                </span>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}

function truncateMiddle(s, max) {
  if (!s || s.length <= max) return s;
  const half = Math.floor((max - 1) / 2);
  return s.slice(0, half) + "…" + s.slice(-half);
}
