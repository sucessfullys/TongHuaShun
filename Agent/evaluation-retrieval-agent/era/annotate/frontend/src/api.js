// Thin fetch wrapper for the /era:annotate API.
//
// All sample_keys may carry slashes (e.g. ``dress/dress_complex_background/
// sample_001``), which is why ``getSample`` and ``putAnnotation`` URL-encode
// each path segment instead of the whole key — preserves the slashes so the
// FastAPI ``{sample_key:path}`` route binds correctly.

async function json(res) {
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`);
  }
  return res.json();
}

const putJSON = (url, body) =>
  fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(json);

/** URL-encode each path segment, leaving slashes intact. */
function encodeSampleKey(key) {
  return key.split("/").map(encodeURIComponent).join("/");
}

export const getOverview = () => fetch("/api/samples").then(json);

export const getHealth = () => fetch("/api/health").then(json);

export const getSample = (key) =>
  fetch(`/api/sample/${encodeSampleKey(key)}`).then(json);

export const putAnnotation = (key, methodId, annotation) =>
  putJSON(`/api/sample/${encodeSampleKey(key)}/annotation`, {
    method_id: methodId,
    annotation,
  });

export const imageUrl = (method, sample, role) =>
  `/api/image?method=${encodeURIComponent(method)}` +
  `&sample=${encodeURIComponent(sample)}&role=${encodeURIComponent(role)}`;
