from langfuse import get_client

langfuse = get_client()

prompt = langfuse.get_prompt(
    "muallim-system-prompt",
    type="chat",
)

print(prompt)

compiled = prompt.compile(
    language="Arabic",
    context="HELLO"
)

print(compiled)