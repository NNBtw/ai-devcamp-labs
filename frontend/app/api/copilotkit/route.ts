import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";
import { NextRequest } from "next/server";

export const runtime = "nodejs";

// next start and the server-side loopback proxy must use the same port.
const frontendPort = process.env.PORT ?? "3000";
if (frontendPort.length < 1 || frontendPort.length > 5 || /[^0-9]/.test(frontendPort) ||
    Number(frontendPort) < 1 || Number(frontendPort) > 65535) {
  throw new Error("Invalid frontend PORT");
}

// Bridges the browser to the ADK backend's AG-UI endpoint.
const copilotRuntime = new CopilotRuntime({
  agents: {
    social_poster: new HttpAgent({
      // Keep ADC in /api/adk; never derive this URL from request headers.
      url: `http://127.0.0.1:${frontendPort}/api/adk`,
    }),
  },
});

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime: copilotRuntime,
    serviceAdapter: new ExperimentalEmptyAdapter(),
    endpoint: "/api/copilotkit",
  });
  return handleRequest(req);
};
