/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The API runs locally per D1. Proxying keeps the browser same-origin, so the
  // app works without CORS in production and without a public API surface.
  async rewrites() {
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/:path*" }];
  },
};
export default nextConfig;
