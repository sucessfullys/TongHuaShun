import React, { useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import { imageUrl } from "./api.js";

const FLAG_FAMILIES = ["A", "hybrid"];

export default function ReviewView({
  data,
  sampleIdx,
  gotoSample,
  visited,
  lookups,
  flagItem,
  markComparison,
  finalized,
}) {
  const sample = data.samples[sampleIdx];
  const total = data.samples.length;
  const configById = useMemo(
    () => Object.fromEntries(data.configs.map((c) => [c.combination_id, c])),
    [data.configs],
  );
  const methodLabels = useMemo(
    () =>
      data.methods.map((m, i) => ({ id: m.method_id, label: `Method ${i + 1}` })),
    [data.methods],
  );

  // Warm the browser cache for the adjacent samples so turning the page is
  // instant — runs on every page turn; out-of-range neighbours are skipped.
  useEffect(() => {
    const prefetch = (idx) => {
      const s = data.samples[idx];
      if (!s) return;
      const urls = [];
      for (const im of (s.input && s.input.images) || []) {
        urls.push(imageUrl(im.method_id, s.sample_key, im.role));
      }
      for (const cell of s.cells || []) {
        urls.push(imageUrl(cell.method_id, s.sample_key, "output"));
      }
      for (const u of urls) {
        const img = new Image();
        img.src = u;
      }
    };
    prefetch(sampleIdx - 1);
    prefetch(sampleIdx + 1);
  }, [sampleIdx, data]);

  if (!sample)
    return <div className="center-msg">No evaluation samples to review.</div>;

  const input = sample.input || {};
  const inputImages = input.images || [];
  const labelSpan = Math.max(inputImages.length, 1);
  const gridCols = labelSpan + data.methods.length;

  return (
    <div>
      <div className="review-hint">
        Each method's result image sits directly above its scores. Flag any
        Family-A / hybrid judgement that looks wrong — anything left unflagged
        counts as <em>correct</em>. For Family B, mark whether its ranking of
        the methods is right.
      </div>

      <div className="nav">
        <button
          className="nav-btn"
          onClick={() => gotoSample(sampleIdx - 1)}
          disabled={sampleIdx === 0}
          aria-label="previous sample"
        >
          ‹
        </button>
        <button
          className="nav-btn"
          onClick={() => gotoSample(sampleIdx + 1)}
          disabled={sampleIdx === total - 1}
          aria-label="next sample"
        >
          ›
        </button>
        <span className="nav-title">{sample.sample_key}</span>
        <span className="nav-count">
          {sampleIdx + 1} / {total}
        </span>
        <div className="progress-wrap">
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${(visited.size / total) * 100}%` }}
            />
          </div>
          <div className="progress-label">
            {visited.size} / {total} samples visited
          </div>
        </div>
      </div>

      {data.task_family === "generation" && input.prompt && (
        <div className="panel prompt-banner">“{input.prompt}”</div>
      )}

      <div
        className="review-grid"
        key={sample.sample_key}
        style={{ "--cols": gridCols, "--label-span": labelSpan }}
      >
        <ImageBand
          sample={sample}
          inputImages={inputImages}
          methodLabels={methodLabels}
        />
        <FamilyABand
          sample={sample}
          methods={data.methods}
          methodLabels={methodLabels}
          configs={data.configs.filter((c) => FLAG_FAMILIES.includes(c.family))}
          lookups={lookups}
          flagItem={flagItem}
          finalized={finalized}
        />
        <FamilyBBand
          sample={sample}
          methods={data.methods}
          methodLabels={methodLabels}
          configById={configById}
          lookups={lookups}
          markComparison={markComparison}
          finalized={finalized}
        />
      </div>

      <div className="kbd-hint">
        <span className="kbd">←</span> <span className="kbd">→</span> change
        sample &nbsp;·&nbsp; <span className="kbd">s</span> save draft
      </div>
    </div>
  );
}

/* ---- image band ----------------------------------------------------- */

function ImageBand({ sample, inputImages, methodLabels }) {
  const labelOf = (mid) =>
    (methodLabels.find((m) => m.id === mid) || {}).label || mid;
  return (
    <div className="img-band">
      {inputImages.length === 0 && <div className="img-cell" />}
      {inputImages.map((im, i) => (
        <motion.div
          className="img-cell"
          key={`in-${im.role}`}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.06, duration: 0.3 }}
        >
          <div className="method-cap">
            <span className="method-cap-label input">input</span>
            <span className="method-cap-id" title={im.role}>
              {im.role}
            </span>
          </div>
          <img
            className="method-img"
            src={imageUrl(im.method_id, sample.sample_key, im.role)}
            alt={im.role}
            title={im.role}
            loading="lazy"
          />
        </motion.div>
      ))}
      {sample.cells.map((cell, i) => (
        <motion.div
          className="img-cell"
          key={cell.method_id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: (inputImages.length + i) * 0.06, duration: 0.3 }}
        >
          <div className="method-cap">
            <span className="method-cap-label">{labelOf(cell.method_id)}</span>
            <span className="method-cap-id" title={cell.method_id}>
              {cell.method_id}
            </span>
          </div>
          <img
            className="method-img"
            src={imageUrl(cell.method_id, sample.sample_key, "output")}
            alt={cell.method_id}
            title={cell.method_id}
            loading="lazy"
          />
        </motion.div>
      ))}
    </div>
  );
}

/* ---- collapsible band shell ----------------------------------------- */

function Band({ title, children }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="band">
      <button className="band-header" onClick={() => setOpen((o) => !o)}>
        <span className="band-arrow">{open ? "▾" : "▸"}</span>
        {title}
      </button>
      {open && children}
    </div>
  );
}

function IntroToggle({ open, onToggle }) {
  return (
    <button className="intro-toggle" onClick={onToggle}>
      <span className="intro-arrow">{open ? "▾" : "▸"}</span>
      what this measures
    </button>
  );
}

function SubHead({ label, methodLabels }) {
  return (
    <div className="band-subhead-row">
      <div className="band-subhead">{label}</div>
      {methodLabels.map((m) => (
        <div className="band-subhead mcol" key={m.id}>
          {m.label}
        </div>
      ))}
    </div>
  );
}

/* ---- Family A + hybrid ---------------------------------------------- */

function FamilyABand({
  sample,
  methods,
  methodLabels,
  configs,
  lookups,
  flagItem,
  finalized,
}) {
  const pivot = useMemo(() => {
    const p = {};
    for (const cell of sample.cells) {
      for (const j of cell.judges) {
        (p[j.combination_id] ||= {})[cell.method_id] = j;
      }
    }
    return p;
  }, [sample]);

  const rows = configs.filter((c) => pivot[c.combination_id]);
  if (rows.length === 0) return null;

  return (
    <Band title="FAMILY A + HYBRID — flag any judgement that is wrong">
      <SubHead label="Evaluator" methodLabels={methodLabels} />
      {rows.map((c) => (
        <FamilyAConfigRow
          key={c.combination_id}
          config={c}
          methods={methods}
          perMethod={pivot[c.combination_id]}
          sampleKey={sample.sample_key}
          lookups={lookups}
          flagItem={flagItem}
          finalized={finalized}
        />
      ))}
    </Band>
  );
}

function FamilyAConfigRow({
  config,
  methods,
  perMethod,
  sampleKey,
  lookups,
  flagItem,
  finalized,
}) {
  const [descOpen, setDescOpen] = useState(false);
  return (
    <>
      <div className="fa-row">
        <div className="fa-label">
          <div className="fa-label-head">
            <span className={"fam " + config.family}>{config.family}</span>
            <span className="fa-config-id">{config.combination_id}</span>
          </div>
          {config.description && (
            <IntroToggle
              open={descOpen}
              onToggle={() => setDescOpen((o) => !o)}
            />
          )}
        </div>
        {methods.map((m) => (
          <JudgeCell
            key={m.method_id}
            judge={perMethod[m.method_id]}
            mark={lookups.item(sampleKey, m.method_id, config.combination_id)}
            finalized={finalized}
            onFlag={(label, comment) =>
              flagItem({
                sample_key: sampleKey,
                method_id: m.method_id,
                combination_id: config.combination_id,
                label,
                comment,
              })
            }
          />
        ))}
      </div>
      {descOpen && config.description && (
        <div className="fa-desc">{config.description}</div>
      )}
    </>
  );
}

function JudgeCell({ judge, mark, finalized, onFlag }) {
  const flagged = !!mark;
  const [comment, setComment] = useState(mark?.comment || "");
  if (!judge) return <div className="fa-cell empty">—</div>;

  const toggle = () => {
    if (finalized) return;
    onFlag(flagged ? "ok" : "wrong", "");
  };

  return (
    <div className={"fa-cell" + (flagged ? " flagged" : "")}>
      <div className="fa-cell-top">
        <span className="fa-score">{judge.display}</span>
        <button
          className={"flag-btn" + (flagged ? " on" : "")}
          onClick={toggle}
          disabled={finalized}
        >
          {flagged ? "✗ wrong" : "flag"}
        </button>
      </div>
      {flagged && (
        <textarea
          className="comment-input"
          placeholder="Why is this judgement wrong?"
          value={comment}
          disabled={finalized}
          onChange={(e) => setComment(e.target.value)}
          onBlur={() => onFlag("wrong", comment)}
        />
      )}
    </div>
  );
}

/* ---- Family B ------------------------------------------------------- */

function FamilyBBand({
  sample,
  methods,
  methodLabels,
  configById,
  lookups,
  markComparison,
  finalized,
}) {
  const rankings = sample.family_b_rankings || [];
  if (rankings.length === 0) return null;
  return (
    <Band title="FAMILY B — is the relative ranking of the methods correct?">
      <SubHead label="Metric" methodLabels={methodLabels} />
      {rankings.map((r) => (
        <FamilyBConfigRow
          key={r.combination_id}
          ranking={r}
          methods={methods}
          description={configById[r.combination_id]?.description}
          mark={lookups.comparison(sample.sample_key, r.combination_id)}
          finalized={finalized}
          onMark={(label, comment) =>
            markComparison({
              sample_key: sample.sample_key,
              combination_id: r.combination_id,
              label,
              comment,
            })
          }
        />
      ))}
    </Band>
  );
}

function FamilyBConfigRow({
  ranking,
  methods,
  description,
  mark,
  finalized,
  onMark,
}) {
  const isWrong = !!mark;
  const [comment, setComment] = useState(mark?.comment || "");
  const [descOpen, setDescOpen] = useState(false);
  const rankOf = (mid) => ranking.ranked_method_ids.indexOf(mid) + 1;

  return (
    <>
      <div className="fb-row">
        <div className="fb-label">
          <div className="fa-label-head">
            <span className="fam B">B</span>
            <span className="fa-config-id">{ranking.combination_id}</span>
          </div>
          <div className="cmp-controls">
            <button
              className={"cmp-btn correct" + (!isWrong ? " on" : "")}
              onClick={() => !finalized && onMark("correct", "")}
              disabled={finalized}
            >
              ✓ correct
            </button>
            <button
              className={"cmp-btn wrong" + (isWrong ? " on" : "")}
              onClick={() => !finalized && onMark("wrong", comment)}
              disabled={finalized}
            >
              ✗ wrong
            </button>
          </div>
          {description && (
            <IntroToggle
              open={descOpen}
              onToggle={() => setDescOpen((o) => !o)}
            />
          )}
          {isWrong && (
            <textarea
              className="comment-input"
              placeholder="What is the correct ranking, and why?"
              value={comment}
              disabled={finalized}
              onChange={(e) => setComment(e.target.value)}
              onBlur={() => onMark("wrong", comment)}
            />
          )}
        </div>
        {methods.map((m) => {
          const r = rankOf(m.method_id);
          return (
            <div className="fb-cell" key={m.method_id}>
              <span className={"rank-badge" + (r === 1 ? " r1" : "")}>
                {r > 0 ? `#${r}` : "—"}
              </span>
              <span className="fb-score">
                {ranking.per_method_display[m.method_id] || "—"}
              </span>
            </div>
          );
        })}
      </div>
      {descOpen && description && <div className="fa-desc">{description}</div>}
    </>
  );
}
