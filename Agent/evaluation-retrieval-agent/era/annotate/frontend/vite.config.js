import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build the SPA into ../static, which the FastAPI app serves. `base: "./"`
// keeps asset references relative so the bundle works behind any mount point.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../static",
    emptyOutDir: true,
  },
  server: {
    // `npm run dev` proxies the API to a locally-running era.cli serve-annotate.
    proxy: { "/api": "http://127.0.0.1:8801" },
  },
});
