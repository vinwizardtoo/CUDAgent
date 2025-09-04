/** @type {import('next').NextConfig} */
// Default to IPv4 to avoid macOS localhost -> ::1 issues
const API_BASE = process.env.API_BASE_URL || 'http://127.0.0.1:8000';

const nextConfig = {
  reactStrictMode: true,
  experimental: {
    typedRoutes: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${API_BASE}/:path*`,
      },
    ];
  },
};

export default nextConfig;
