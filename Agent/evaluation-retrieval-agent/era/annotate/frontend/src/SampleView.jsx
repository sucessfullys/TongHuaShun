import React, { useMemo } from "react";
import { motion } from "motion/react";
import { imageUrl } from "./api.js";
import MethodNotesCell from "./MethodNotesCell.jsx";

/** The heart of the UI: a Stage-8-style aligned grid.
 *
 *  Two bands sharing one column count:
 *   - Image band: one cell per input role (shared across methods) +
 *     one cell per method (its tryon_result image).
 *   - Notes band: input columns hold "shared input" stubs; method
 *     columns hold per-method notes textareas. Columns line up
 *     vertically so each method's image sits directly above its
 *     own textarea.
 */
export default function SampleView({
  current, overview, saveAnnotation, savedTick,
}) {
  const inputRoles = overview.input_roles || [];
  const methods = current.methods || [];
  const totalCols = inputRoles.length + methods.length;

  // Build the input-image cells. Inputs are shared across methods, so
  // we pick the first method that has the role to source the image.
  const inputImageCells = inputRoles.map((role) => {
    const sourceMethod = methods.find(
      (m) => m.present && m.roles && m.roles[role],
    );
    return {
      key: `input-${role}`,
      role,
      url: sourceMethod
        ? imageUrl(sourceMethod.method_id, current.sample_key, role)
        : null,
    };
  });

  const methodImageCells = methods.map((m) => {
    const outputRole = overview.output_role || "output";
    const url = m.present && m.roles && m.roles[outputRole]
      ? imageUrl(m.method_id, current.sample_key, outputRole)
      : null;
    return { method: m, url };
  });

  const gridStyle = useMemo(
    () => ({ "--cols": Math.max(1, totalCols) }),
    [totalCols],
  );

  return (
    <section className="annotate-grid" style={gridStyle}>
      <div className="prompt-banner">
        Sample <span className="mono">{current.sample_key}</span>
      </div>

      {/* ---- image band ---- */}
      <div className="img-band">
        {inputImageCells.map((c, i) => (
          <motion.div
            key={c.key}
            className="img-cell input"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.04, duration: 0.25 }}
          >
            <div className="method-cap">
              <span className="method-cap-label input">{c.role}</span>
              <span className="method-cap-id">shared input</span>
            </div>
            {c.url ? (
              <img className="method-img" src={c.url} alt={c.role} loading="lazy" />
            ) : (
              <div className="img-missing">missing</div>
            )}
          </motion.div>
        ))}
        {methodImageCells.map((c, i) => (
          <motion.div
            key={`m-${c.method.method_id}`}
            className="img-cell method"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              delay: (inputImageCells.length + i) * 0.04,
              duration: 0.25,
            }}
          >
            <div className="method-cap">
              <span className="method-cap-label">tryon_result</span>
              <span className="method-cap-id" title={c.method.method_id}>
                {c.method.method_id}
              </span>
            </div>
            {c.url ? (
              <img
                className="method-img"
                src={c.url}
                alt={`${c.method.method_id} result`}
                loading="lazy"
              />
            ) : (
              <div className="img-missing">missing</div>
            )}
          </motion.div>
        ))}
      </div>

      {/* ---- notes band (column-aligned with the image band) ---- */}
      <div className="notes-band">
        {inputRoles.map((role) => (
          <div key={`stub-${role}`} className="notes-cell input-stub">
            (shared input — no notes)
          </div>
        ))}
        {methods.map((m) => (
          <MethodNotesCell
            key={`notes-${m.method_id}`}
            sampleKey={current.sample_key}
            methodId={m.method_id}
            initialText={(current.per_method || {})[m.method_id] || ""}
            disabled={!m.present}
            saveAnnotation={saveAnnotation}
            savedTick={savedTick}
          />
        ))}
      </div>

      <p className="kbd-hint">
        <span className="kbd">←</span> / <span className="kbd">→</span>{" "}
        navigate samples · <span className="kbd">Ctrl+S</span> save all dirty
        notes · empty notes are fine, just leave the box blank
      </p>
    </section>
  );
}
