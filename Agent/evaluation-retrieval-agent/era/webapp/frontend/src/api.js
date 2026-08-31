// Thin fetch wrapper for the ERA Stage 8 feedback API.

async function json(res) {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

const putJSON = (url, body) =>
  fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(json);

export const getReview = () => fetch("/api/review").then(json);
export const getOverview = () => fetch("/api/overview").then(json);

export const putItem = (mark) => putJSON("/api/feedback/item", mark);
export const putComparison = (mark) => putJSON("/api/feedback/comparison", mark);
export const putGeneral = (text) => putJSON("/api/feedback/general", { text });

export const saveDraft = () =>
  fetch("/api/feedback/save", { method: "POST" }).then(json);

export async function finalize(reopen = false) {
  const res = await fetch("/api/feedback/finalize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reopen }),
  });
  const body = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, body };
}

export const imageUrl = (method, sample, role) =>
  `/api/image?method=${encodeURIComponent(method)}` +
  `&sample=${encodeURIComponent(sample)}&role=${encodeURIComponent(role)}`;
