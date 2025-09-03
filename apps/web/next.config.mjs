/** @type {import('next').NextConfig} */
const API_BASE = process.env.API_BASE_URL || 'http://localhost:8000';

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
