import { proxyBackend } from "../../lib/backend-proxy";

export const runtime = "nodejs";

export async function POST(request: Request) {
  return proxyBackend(request, "/api/adk");
}
