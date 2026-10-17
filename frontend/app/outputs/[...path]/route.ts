import { proxyBackend } from "../../lib/backend-proxy";

export const runtime = "nodejs";

export async function GET(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  return proxyBackend(request, "/outputs/" + path.map(encodeURIComponent).join("/"));
}
