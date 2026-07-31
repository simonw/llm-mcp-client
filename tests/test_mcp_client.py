import base64

import llm
import pytest
from mcp import types
from mcp.server import MCPServer

import llm_mcp_client
from llm_mcp_client import MCP, MCPToolError, _convert_result


@pytest.fixture
def server():
    server = MCPServer("demo")

    @server.tool()
    def add(a: int, b: int) -> int:
        "Add two numbers together."
        return a + b

    @server.tool()
    def shout(text: str) -> str:
        "Convert text to uppercase."
        return text.upper()

    return server


def test_plugin_is_installed():
    assert llm_mcp_client.MCP is MCP


def test_register_tools_hook():
    registered = {}

    def register(tool, name=None):
        registered[name or tool.__name__] = tool

    llm_mcp_client.register_tools(register)
    assert registered == {"MCP": MCP}


def test_tools_are_discovered_from_server(server):
    toolbox = MCP(server)
    tools = {tool.name: tool for tool in toolbox.tools()}
    assert set(tools) == {"add", "shout"}
    add = tools["add"]
    assert add.description == "Add two numbers together."
    assert add.plugin == "llm_mcp_client"
    assert set(add.input_schema["properties"]) == {"a", "b"}
    assert add.input_schema["properties"]["a"]["type"] == "integer"


def test_prefix_is_applied_to_tool_names(server):
    toolbox = MCP(server, prefix="demo_")
    names = {tool.name for tool in toolbox.tools()}
    assert names == {"demo_add", "demo_shout"}


def test_tool_defs_are_cached(server):
    toolbox = MCP(server)
    first = toolbox.tool_defs()
    assert toolbox.tool_defs() is first
    toolbox.refresh()
    assert toolbox.tool_defs() is not first


def test_calling_a_tool(server):
    toolbox = MCP(server)
    tools = {tool.name: tool for tool in toolbox.tools()}
    result = llm_mcp_client._run(tools["add"].implementation(a=3, b=4))
    assert result == "7"
    result = llm_mcp_client._run(tools["shout"].implementation(text="hello"))
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_calling_a_tool_from_async_context(server):
    toolbox = MCP(server)
    tools = {tool.name: tool for tool in toolbox.tools()}
    result = await tools["add"].implementation(a=1, b=2)
    assert result == "3"


def test_error_result_raises():
    result = types.CallToolResult(
        content=[types.TextContent(type="text", text="Something went wrong")],
        isError=True,
    )
    with pytest.raises(MCPToolError, match="Something went wrong"):
        _convert_result(result)


def test_image_content_becomes_attachment():
    png = b"\x89PNG fake image bytes"
    result = types.CallToolResult(
        content=[
            types.TextContent(type="text", text="Here is an image"),
            types.ImageContent(
                type="image",
                data=base64.b64encode(png).decode("ascii"),
                mimeType="image/png",
            ),
        ],
    )
    output = _convert_result(result)
    assert isinstance(output, llm.ToolOutput)
    assert output.output == "Here is an image"
    assert len(output.attachments) == 1
    attachment = output.attachments[0]
    assert attachment.content == png
    assert attachment.type == "image/png"


def test_structured_content_used_when_no_text():
    result = types.CallToolResult(
        content=[],
        structuredContent={"answer": 42},
    )
    assert _convert_result(result) == '{"answer": 42}'


def test_invalid_mode_rejected(server):
    with pytest.raises(ValueError, match="mode must be one of"):
        MCP(server, mode="bogus")


def test_helper_methods_are_not_exposed_as_tools(server):
    toolbox = MCP(server)
    names = {tool.name for tool in toolbox.tools()}
    assert "MCP_tool_defs" not in names
    assert "MCP_refresh" not in names
    method_tool_names = {tool.name for tool in MCP.method_tools()}
    assert "MCP_tool_defs" not in method_tool_names
    assert "MCP_refresh" not in method_tool_names
