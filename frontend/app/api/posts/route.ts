import { proxyBackend } from "../../lib/backend-proxy";

export const runtime = "nodejs";

export async function GET(request: Request) {
  return proxyBackend(request, "/api/posts");
}
