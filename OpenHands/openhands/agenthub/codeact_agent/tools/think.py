# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_THINK_DESCRIPTION = """Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it ONLY when complex reasoning or brainstorming is genuinely needed before a critical decision.

IMPORTANT: Each think call counts as one iteration toward your limit. Do NOT think routinely between every action. Prefer taking concrete actions (search, read, edit) directly. Reserve think for:
1. After discovering a bug's root cause, to briefly compare 2-3 fix strategies before choosing one.
2. After multiple failed attempts, to step back and reassess the approach.

Do NOT use think:
- Right after reading a file (just proceed to the next action)
- Right after a search (just act on the results)
- To summarize what you just observed (the observation is already in context)
- Between every step as a habit

The tool simply logs your thought process for better transparency and does not execute any code or make changes."""

ThinkTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='think',
        description=_THINK_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'thought': {'type': 'string', 'description': 'The thought to log.'},
            },
            'required': ['thought'],
        },
    ),
)
