"""LLM plugin that exposes tools from MCP servers as LLM tools.

Uses the ``mcp`` Python SDK (the same library as mcp-explorer) to connect
to an MCP server, discover its tools and make them available to LLM as a
Toolbox called ``MCP``:

    llm -T 'MCP("https://example.com/mcp")' 'Ask something' --td
"""

import asyncio
import base64
import json
import threading

import llm
from mcp import types
from mcp.client import Client
from mcp.types.version import LATEST_MODERN_VERSION

MODES = ("auto", "stateless", "legacy")


class MCPToolError(Exception):
    "Raised when an MCP server returns an error result for a tool call."


def _run(coro):
    "Run a coroutine to completion, even from inside a running event loop."
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside a loop - run the coroutine on its own loop in a thread
    outcome = {}

    def runner():
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as ex:
            outcome["error"] = ex

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _convert_result(result: types.CallToolResult):
    "Turn a CallToolResult into a return value for an LLM tool."
    text_parts = []
    attachments = []

    for content in result.content:
        if isinstance(content, types.TextContent):
            text_parts.append(content.text)
        elif isinstance(content, (types.ImageContent, types.AudioContent)):
            attachments.append(
                llm.Attachment(
                    content=base64.b64decode(content.data),
                    type=content.mime_type,
                )
            )
        elif isinstance(content, types.ResourceLink):
            text_parts.append(f"[resource: {content.name} ({content.uri})]")
        elif isinstance(content, types.EmbeddedResource):
            resource = content.resource
            if isinstance(resource, types.TextResourceContents):
                text_parts.append(resource.text)
            else:
                attachments.append(
                    llm.Attachment(
                        content=base64.b64decode(resource.blob),
                        type=resource.mime_type or "application/octet-stream",
                    )
                )

    output = "\n".join(text_parts)

    if result.is_error:
        raise MCPToolError(output or "MCP tool call failed")

    if not output and result.structured_content is not None:
        output = json.dumps(result.structured_content, default=repr)

    if attachments:
        return llm.ToolOutput(output=output, attachments=attachments)
    return output


class MCP(llm.Toolbox):
    """Expose the tools from an MCP server as LLM tools.

    Usage:

        MCP("https://example.com/mcp")
        MCP("https://example.com/mcp", mode="legacy", prefix="demo_")

    - server: URL of the MCP server (or anything else accepted by
      mcp.client.Client, e.g. an in-process server instance)
    - mode: "auto" (default) negotiates the protocol, "stateless" forces
      modern stateless MCP, "legacy" forces the initialize handshake
    - prefix: optional string prepended to each tool name, useful to
      avoid clashes when using tools from more than one server
    """

    # Keep helper methods from being exposed as tools themselves
    _blocked = llm.Toolbox._blocked + ("tool_defs", "refresh")

    def __init__(self, server, mode="auto", prefix=""):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        self.server = server
        self.mode = mode
        self.prefix = prefix
        self._tool_defs = None
        self._lock = threading.Lock()

    def _client(self):
        client_mode = {
            "auto": "auto",
            "stateless": LATEST_MODERN_VERSION,
            "legacy": "legacy",
        }[self.mode]
        return Client(self.server, mode=client_mode)

    async def _fetch_tool_defs(self):
        tool_defs = []
        cursor = None
        async with self._client() as client:
            while True:
                result = await client.list_tools(cursor=cursor)
                tool_defs.extend(result.tools)
                cursor = result.next_cursor
                if not cursor:
                    return tool_defs

    async def _call_tool(self, name, arguments):
        async with self._client() as client:
            result = await client.call_tool(name, arguments)
        return _convert_result(result)

    def _make_implementation(self, name):
        async def implementation(**kwargs):
            return await self._call_tool(name, kwargs)

        implementation.__name__ = name
        return implementation

    def tool_defs(self):
        "The mcp.types.Tool definitions fetched from the server, cached."
        with self._lock:
            if self._tool_defs is None:
                self._tool_defs = _run(self._fetch_tool_defs())
            return self._tool_defs

    def refresh(self):
        "Discard cached tool definitions so the next use re-fetches them."
        with self._lock:
            self._tool_defs = None

    def tools(self):
        for tool_def in self.tool_defs():
            yield llm.Tool(
                name=self.prefix + tool_def.name,
                description=tool_def.description,
                input_schema=tool_def.input_schema
                or {"type": "object", "properties": {}},
                implementation=self._make_implementation(tool_def.name),
                plugin="llm_mcp_client",
            )
        yield from self._extra_tools


@llm.hookimpl
def register_tools(register):
    register(MCP)
