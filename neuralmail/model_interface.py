import logging
import json
import openai
import tiktoken
from typing import Dict, List, Any, Optional, Tuple
from .utils import openai_api_client

logger = logging.getLogger(__name__)


class ModelInterface:
    """
    Handles communication with the LLM model for the research agent.
    """

    def __init__(
        self,
        api_key: str,
        model: str = None,
        base_url: str = None,
        embedding_api_key: str = None,
        embedding_base_url: str = None,
    ):
        """
        Initialize the model interface.

        Args:
            api_key: OpenAI API key
            model: Model to use for research
            base_url: Custom base URL for OpenAI-compatible providers
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        self.client = self._create_client()
        self.embedding_client = (
            openai.OpenAI(
                api_key=self.embedding_api_key,
                base_url=(
                    self.embedding_base_url
                    if self.embedding_base_url
                    else "https://api.openai.com/v1"
                ),
            )
            if self.embedding_api_key
            else self.client
        )

        # Handle tokenization for different model providers
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for non-OpenAI models (like Google Gemini)
            # Use cl100k_base encoding as a reasonable default
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.warning(
                f"Model '{model}' not found in tiktoken, using cl100k_base encoding as fallback"
            )

        self.max_tokens = (
            256000
        )
        self.token_usage_history = []  # Track token usage across calls

    def _create_client(self):
        """Create and return an OpenAI client."""
        return openai_api_client(self.api_key, self.base_url)

    def format_tools_for_prompt(self, tools: Dict[str, Dict]) -> str:
        """
        Format the available tools for the prompt.

        Args:
            tools: Dictionary of tool definitions

        Returns:
            Formatted tools documentation
        """
        tools_doc = "Available Tools:\n\n"

        for tool_name, tool_info in tools.items():
            tools_doc += f"- {tool_name}({', '.join(tool_info['params'])})\n"
            tools_doc += f"  Description: {tool_info['description']}\n"
            if "param_descriptions" in tool_info:
                tools_doc += "  Parameters:\n"
                for param, desc in tool_info["param_descriptions"].items():
                    tools_doc += f"    - {param}: {desc}\n"
            tools_doc += "\n"

        return tools_doc

    def generate_system_prompt(self, tools: Dict[str, Dict]) -> str:
        """
        Generate the system prompt for the research agent.

        Args:
            tools: Dictionary of available tools

        Returns:
            Formatted system prompt
        """
        tools_doc = self.format_tools_for_prompt(tools)

        system_prompt = f"""You are an email research assistant. Your goal is to answer the user's query by iteratively searching their emails and using other tools.

{tools_doc}

Your task is to perform multiple calls to the available tools to gather information. For each call, you will formulate a new query or action based on the insights gathered from previous results, which are stored in the scratchpad.

**Workflow:**
1.  **Initial Search**: Start with a broad search related to the query to get a general overview.
2.  **Iterative Refinement**: For each subsequent step, analyze the information in the scratchpad and formulate a new, more specific action to explore a particular aspect related to the user's query.
3.  **Final Response**: Once you have gathered enough information, use the `respond` tool to provide a comprehensive answer to the user's query.

Always format your tool calls as valid JSON objects. For example:
```json
{{"tool": "search_by_embedding", "params": {{"query": "project deadline", "top_k": 50}}}}
```
"""
        return system_prompt

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoding.encode(text))

    def _truncate_context(
        self, context: List[Dict[str, Any]], max_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        Truncate context history to fit within token limit while preserving most recent entries.

        Args:
            context: List of context entries
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated context list
        """
        if not context:
            return []

        # Start with most recent entries
        truncated = []
        current_tokens = 0

        for entry in reversed(context):
            # Convert entry to string representation
            entry_str = f"\nStep {len(truncated)+1}:\n"
            if "tool" in entry:
                entry_str += f"Tool: {entry['tool']}\n"
                entry_str += f"Parameters: {json.dumps(entry['params'])}\n"
            if "result" in entry:
                # Only truncate extremely long results (>20000 chars)
                result = entry["result"]
                if isinstance(result, str) and len(result) > 20000:
                    # Keep a much larger portion when truncating
                    result = (
                        result[:15000]
                        + "... [truncated "
                        + str(len(result) - 15000)
                        + " chars]"
                        + result[-5000:]
                    )
                entry_str += f"Result: {result}\n"

            # Check if adding this entry would exceed the limit
            entry_tokens = self._count_tokens(entry_str)
            if current_tokens + entry_tokens > max_tokens:
                break

            current_tokens += entry_tokens
            truncated.insert(0, entry)  # Insert at start to maintain order

        if truncated:
            logger.info(
                f"Truncated context from {len(context)} to {len(truncated)} entries to fit token limit"
            )
        return truncated

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context history into a string, with length monitoring."""
        if not context:
            return " No previous steps yet."

        context_str = ""
        for i, entry in enumerate(context):
            context_str += f"\nStep {i+1}:\n"
            if "tool" in entry:
                context_str += f"Tool: {entry['tool']}\n"
                context_str += f"Parameters: {json.dumps(entry['params'])}\n"
            if "result" in entry:
                result = entry["result"]
                if isinstance(result, str) and len(result) > 20000:
                    # For extremely long results, preserve more meaningful content
                    beginning = 15000  # Keep first 15000 chars
                    ending = 5000  # Keep last 5000 chars
                    truncated_length = len(result) - (beginning + ending)
                    result = (
                        result[:beginning]
                        + f"\n... [truncated {truncated_length} characters] ...\n"
                        + result[-ending:]
                    )
                context_str += f"Result: {result}\n"

        return context_str

    def call_model(
        self,
        user_query: str,
        tools: Dict[str, Dict],
        context: List[Dict[str, Any]],
        scratch_pad: str,
        current_iteration: int,
        max_iterations: int,
        research_tools=None,  # Add parameter to receive ResearchTools instance
    ) -> Tuple[str, Optional[Dict]]:
        """
        Call the OpenAI model to get the next research step.

        Args:
            user_query: The original user query
            tools: Dictionary of available tools
            context: List of previous actions and results
            scratch_pad: Current scratch pad content
            current_iteration: Current iteration number
            max_iterations: Maximum number of iterations
            research_tools: ResearchTools instance for logging

        Returns:
            Tuple of (raw_response, parsed_tool_call)
        """
        system_prompt = self.generate_system_prompt(tools)
        system_tokens = self._count_tokens(system_prompt)

        # Calculate available tokens for dynamic content - utilizing much more of the context window
        available_tokens = (
            self.max_tokens - system_tokens - 2000
        )  # Reserve tokens for response

        # Allocate tokens between context and scratch pad - prioritize scratch pad more now
        # since it contains the accumulated relevant information
        context_allocation = int(available_tokens * 0.6)  # 60% for context
        scratch_pad_allocation = int(available_tokens * 0.4)  # 40% for scratch pad

        # Store the complete data for HTML logging
        complete_context_str = self._format_context(context)
        complete_scratch_pad = scratch_pad

        # Truncate context if needed
        truncated_context = self._truncate_context(context, context_allocation)
        context_str = self._format_context(truncated_context)

        # Truncate scratch pad if needed
        truncated_scratch_pad = scratch_pad
        if scratch_pad and self._count_tokens(scratch_pad) > scratch_pad_allocation:
            # When truncating scratch pad, preserve more content
            chars_per_token = 4  # Approximate chars to tokens
            truncate_at = scratch_pad_allocation * chars_per_token
            # Keep the beginning and end portions
            beginning = (
                truncate_at * 3 // 4
            )  # 3/4 of the allowed content from beginning
            ending = truncate_at // 4  # 1/4 of the allowed content from ending
            truncated_scratch_pad = (
                scratch_pad[:beginning]
                + "\n... [truncated "
                + str(len(scratch_pad) - truncate_at)
                + " chars] ...\n"
                + scratch_pad[-ending:]
            )

        # Format the user message for model - emphasize the scratch pad more now
        user_message_for_model = f"""ORIGINAL QUERY: {user_query}

CURRENT ITERATION: {current_iteration}/{max_iterations}

SCRATCH PAD (Key Research Findings):
{truncated_scratch_pad if truncated_scratch_pad else 'The scratch pad is empty. Start with a broad query.'}

Based on the information in the scratch pad, formulate the next query for the `search_by_embedding` tool.
Your query should be designed to uncover new information and build upon the existing findings.

Respond with a JSON object specifying the tool and parameters."""

        # Create a complete user message for HTML logging
        complete_user_message = f"""ORIGINAL QUERY: {user_query}

CURRENT ITERATION: {current_iteration}/{max_iterations}

RESEARCH CONTEXT (Previous Actions):
{complete_context_str}

SCRATCH PAD (Key Research Findings):
{complete_scratch_pad if complete_scratch_pad else 'The scratch pad is empty. Use search tools to gather information.'}

Based on the information so far (especially the findings in the scratch pad), what tool should you use next? 
If you have gathered enough relevant information in the scratch pad to answer the query, use the respond tool.

Respond with a JSON object specifying the tool and parameters."""

        try:
            # Call the model
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message_for_model},
                ],
            )

            # Track token usage
            usage = response.usage
            self.token_usage_history.append(
                {
                    "total_tokens": usage.total_tokens,
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "iteration": current_iteration,
                }
            )

            # Log token usage
            logger.info(
                f"Token usage - Total: {usage.total_tokens}, "
                f"Prompt: {usage.prompt_tokens}, "
                f"Completion: {usage.completion_tokens}"
            )

            # Extract the response content
            response_content = response.choices[0].message.content

            # Try to parse a JSON tool call from the response
            tool_call = self._extract_tool_call(response_content)

            # Log the LLM call to HTML if research_tools is provided
            if research_tools:
                research_tools.log_llm_call(
                    system_prompt=system_prompt,
                    user_message=complete_user_message,  # Use complete message for HTML logging
                    response_content=response_content,
                    tool_call=tool_call,
                )

            return response_content, tool_call

        except openai.BadRequestError as e:
            if "maximum context length" in str(e).lower():
                logger.warning(
                    "Context length exceeded, reducing context for next iteration"
                )
                # Reduce max_tokens for future calls
                self.max_tokens = int(self.max_tokens * 0.8)
            logger.error(f"OpenAI API error: {e}")

            # Log error to HTML if research_tools is provided
            if research_tools:
                research_tools.log_llm_call(
                    system_prompt=system_prompt,
                    user_message=complete_user_message,  # Use complete message for HTML logging
                    response_content=f"Error: {str(e)}",
                    tool_call=None,
                )

            return f"Error: {str(e)}", None
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")

            # Log error to HTML if research_tools is provided
            if research_tools:
                research_tools.log_llm_call(
                    system_prompt=system_prompt,
                    user_message=complete_user_message,  # Use complete message for HTML logging
                    response_content=f"Error: {str(e)}",
                    tool_call=None,
                )

            return f"Error: {str(e)}", None

    def _extract_tool_call(self, response: str) -> Optional[Dict]:
        """
        Extract a tool call from the model response.

        Args:
            response: Raw model response

        Returns:
            Parsed tool call or None if not found/valid
        """
        # Try to find JSON in the response using different patterns
        json_patterns = [
            r"```json\s*(.*?)\s*```",  # Code block with json tag
            r"```\s*(.*?)\s*```",  # Any code block
            r"{.*}",  # Any JSON object
        ]

        import re

        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    # Try to parse the JSON
                    tool_call = json.loads(match)
                    # Validate that it has the required fields
                    if "tool" in tool_call and isinstance(tool_call["tool"], str):
                        return tool_call
                except Exception:
                    continue

        # If no JSON found with patterns, try to parse the entire response
        try:
            tool_call = json.loads(response)
            if "tool" in tool_call and isinstance(tool_call["tool"], str):
                return tool_call
        except Exception:
            pass

        # No valid tool call found
        return None
