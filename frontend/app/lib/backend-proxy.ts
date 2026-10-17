// Server-side only: imported by Next.js route handlers, never client components.
import { GoogleAuth } from "google-auth-library";

const auth = new GoogleAuth({
  scopes: ["https://www.googleapis.com/auth/cloud-platform"],
});
const GOOGLE_ORIGIN = "https://us-central1-aiplatform.googleapis.com";

function backendBase(): URL {
  const base = new URL(process.env.ADK_BACKEND_ORIGIN || "http://localhost:8000");
  if (base.username || base.password || base.search || base.hash) {
    throw new Error("Invalid backend URL");
  }
  const local = base.protocol === "http:" &&
    ["localhost", "127.0.0.1", "[::1]"].includes(base.hostname);
  if (base.origin !== GOOGLE_ORIGIN && !local) {
    throw new Error("Unsupported backend origin");
  }
  base.pathname = base.pathname.replace(/\/+$/, "");
  return base;
}

function allowedPath(path: string): boolean {
  if (path === "/api/adk" || path === "/api/posts") return true;
  if (!path.startsWith("/outputs/")) return false;
  try {
    return path.slice("/outputs/".length).split("/").every((part) => {
      const decoded = decodeURIComponent(part);
      return decoded.length > 0 && decoded !== "." && decoded !== ".." &&
        !/[\\/\u0000-\u001f\u007f?#]/.test(decoded);
    });
  } catch {
    return false;
  }
}

export async function proxyBackend(request: Request, path: string): Promise<Response> {
  if (!allowedPath(path)) {
    return Response.json({ error: "Invalid backend path" }, { status: 400 });
  }
  try {
    const base = backendBase();
    const target = new URL(base.toString().replace(/\/$/, "") + path);
    const headers = new Headers();
    // Do not forward browser credentials, cookies, Host, or hop-by-hop headers.
    for (const name of ["accept", "content-type", "last-event-id"]) {
      const value = request.headers.get(name);
      if (value !== null) headers.set(name, value);
    }
    if (target.origin === GOOGLE_ORIGIN) {
      const token = await auth.getAccessToken();
      if (!token) throw new Error("ADC did not provide an access token");
      headers.set("authorization", `Bearer ${token}`);
    }
    const upstream = await fetch(target, {
      method: request.method,
      headers,
      body: request.method === "GET" || request.method === "HEAD"
        ? undefined : await request.arrayBuffer(),
      signal: request.signal,
      cache: "no-store",
      // Never follow a redirect with server credentials.
      redirect: "manual",
    });
    if (upstream.status >= 300 && upstream.status < 400 && upstream.status !== 304) {
      await upstream.body?.cancel();
      throw new Error("Backend redirect refused");
    }
    const responseHeaders = new Headers({ "cache-control": "no-store" });
    for (const name of ["content-type", "etag", "last-modified", "content-disposition"]) {
      const value = upstream.headers.get(name);
      if (value !== null) responseHeaders.set(name, value);
    }
    responseHeaders.set("x-accel-buffering", "no");
    // Keep SSE streaming; fetch already handles any upstream decompression.
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: responseHeaders,
    });
  } catch {
    // Do not serialize ADC errors, request headers, or credential details.
    return Response.json({ error: "Backend proxy request failed" }, {
      status: 502,
      headers: { "cache-control": "no-store" },
    });
  }
}
