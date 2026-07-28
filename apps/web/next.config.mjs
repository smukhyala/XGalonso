/** @type {import('next').NextConfig} */

// Where the decision API listens. Overridable so a second instance can be run
// against a branch without editing this file — a worktree serving a change and
// the main checkout serving `main` need different ports, and hand-editing a
// committed config to do it invites the edit being committed by accident.
const API_ORIGIN = process.env.XG_API_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig = {
  reactStrictMode: true,
  // The API runs locally per D1. Proxying keeps the browser same-origin, so the
  // app works without CORS in production and without a public API surface.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_ORIGIN}/:path*` }];
  },
};
export default nextConfig;
