# test_imports.py
from state import AppState, FileStateCache, TaskState, ToolPermissionContext
from tool import Tool, ToolUseContext, ToolResult, build_tool, to_openai_schema
from system_context import get_system_context, get_user_context
from messages import create_user_message, to_openai
from registry import register, find, all_tools

print("✅ All imports OK")

# Verify ToolUseContext defaults work
ctx = ToolUseContext()
print(f"✅ ToolUseContext.get_app_state() = {ctx.get_app_state()}")

# Verify state mutators
state = AppState()
state = __import__('state').add_message(state, create_user_message("hi"))
print(f"✅ State after add_message: {len(state.messages)} messages")

# Verify the bug fix
from pydantic import BaseModel
class In(BaseModel):
    x: int = 0

async def call(args, ctx, can_use_tool=None, parent_msg=None, on_progress=None):
    return ToolResult(data={"y": args.x * 2})

try:
    build_tool(
        name="X", description="x", input_schema=In, call=call,
        prompt="x", max_result_size_chars=1000,
        map_tool_result_to_tool_result_block_param=lambda o, i: {"content": str(o)},
        render_tool_use_message=lambda i, o: "x",
        is_read_only=True,
    )
    print("❌ BUG: is_read_only=True was accepted")
except TypeError as e:
    print(f"✅ Correctly rejected: {e}")

# Verify system context
print(f"✅ System context: {list(get_system_context().keys())}")
print(f"✅ User context: {list(get_user_context().keys())}")