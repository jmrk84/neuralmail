import logging
from typing import Dict, List, Any, Optional
from .model_interface import ModelInterface
from .research_tools import ResearchTools
import json
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    Agent that coordinates the research process using available tools.
    """

    def __init__(
        self,
        api_key: str,
        db_paths: List[str],
        max_iterations: int = 10,
        progress_callback=None,
        base_url: str = None,
        model: str = None,
        embedding_api_key: str = None,
        embedding_base_url: str = None,
    ):
        """
        Initialize the research agent.

        Args:
            api_key: OpenAI API key
            db_paths: List of paths to the SQLite databases
            max_iterations: Maximum number of research iterations
            progress_callback: Optional callback function to report progress
            base_url: Custom base URL for OpenAI-compatible providers
            model: Model to use for research (default: gpt-4o)
        """
        self.model = ModelInterface(
            api_key, model, base_url, embedding_api_key, embedding_base_url
        )
        # Create OpenAI client and pass it to ResearchTools
        self.openai_client = self.model.client
        self.tools = ResearchTools(
            db_paths, self.openai_client, embedding_api_key, embedding_base_url
        )
        self.max_iterations = max_iterations
        self.context = []  # Store research steps and results
        self.scratch_pad = ""  # Store intermediate findings
        self.current_query = None
        self.html_log = []
        self.log_dir = Path("research_logs")
        self.log_dir.mkdir(exist_ok=True)
        self._original_methods = {}  # Store original method implementations
        self.progress_callback = progress_callback  # Add progress callback
        self.key_emails = set()  # Store email IDs that contributed to the research
        logger.info(
            "Research agent initialized with expanded context window capabilities"
        )

    def generate_parallel_queries(
        self, query: str, scratch_pad_content: str, num_queries: int = 5
    ) -> List[str]:
        """
        Generate multiple parallel search queries based on the initial query and scratch pad.
        """
        self.progress_callback(f"Generating {num_queries} parallel queries...")

        prompt = f"""You are a research planner. Your task is to generate {num_queries} diverse and specific sub-queries based on an initial user query and the current research findings. These sub-queries will be executed in parallel to explore different facets of the topic.

USER QUERY: "{query}"

CURRENT RESEARCH FINDINGS (from scratch pad):
{scratch_pad_content}

Instructions:
1. Generate {num_queries} distinct search queries.
2. Each query should be specific and target a different aspect of the original query.
3. The queries should be designed to be executed by a semantic search engine against a corpus of emails.
4. Return the queries as a JSON list of strings. For example: ["query 1", "query 2", ...].
5. Do NOT include any other text or explanation in your response, only the JSON list.

JSON list of {num_queries} search queries:"""

        try:
            # Using response_format for reliable JSON output
            response = self.model.client.chat.completions.create(
                model=self.model.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            queries_str = response.choices[0].message.content
            # The response should be a JSON object, e.g., {"queries": ["q1", "q2"]}
            data = json.loads(queries_str)

            queries = []
            if isinstance(data, list):
                queries = data
            elif isinstance(data, dict):
                # Look for a list of strings in the dictionary's values
                for key, value in data.items():
                    if isinstance(value, list) and all(
                        isinstance(i, str) for i in value
                    ):
                        queries = value
                        break

            if not queries:
                raise ValueError(
                    f"JSON response from model did not contain a list of queries. Got: {data}"
                )

            logger.info(f"Generated {len(queries)} parallel queries: {queries}")
            self.progress_callback(
                f"Generated {len(queries)} parallel queries to explore the topic."
            )
            return queries[:num_queries]
        except Exception as e:
            logger.error(f"Error generating parallel queries: {e}")
            logger.error(traceback.format_exc())
            # Fallback to just the original query in case of failure
            return [query]

    def summarize_results(self, query: str, parallel_results: List[str]) -> str:
        """
        Summarizes a list of search results into a concise summary for the scratch pad.
        """
        self.progress_callback("Consolidating parallel search results...")

        # Filter out empty results before joining
        results_text = "\n\n---\n\n".join(
            filter(None, [str(r) for r in parallel_results])
        )

        if not results_text.strip():
            logger.warning("No parallel results to summarize.")
            return ""

        prompt = f"""You are an information consolidation specialist. Your task is to analyze a set of parallel search results and create a single, concise, and structured summary of the key findings.

ORIGINAL USER QUERY: "{query}"

PARALLEL SEARCH RESULTS (each item may include a field like "citation_id": "<db_name>:<uid>"):
---
{results_text}
---

Instructions:
1. Synthesize all the provided search results.
2. Extract the most important and relevant facts, names, dates, and topics.
3. Organize the information into a structured summary using Markdown (headers, bullet points).
4. Remove redundant information.
5. For any fact you include that comes from a particular result item with a citation_id, append the exact token [[CID:<db_name>:<uid>]] at the end of the relevant sentence or bullet.
6. If multiple items support the same sentence, append multiple tokens one after another with no commas and no extra brackets, e.g., [[CID:a]][[CID:b]] — do NOT write [[CID:a], [CID:b]].
7. Preserve the [[CID:...]] tokens exactly; do not alter their format and do not invent new IDs.
8. The summary will be added to a scratch pad for further research, so keep it dense and concise.

CONSOLIDATED SUMMARY (with inline [[CID:...]] where applicable):"""

        try:
            response = self.model.client.chat.completions.create(
                model=self.model.model, messages=[{"role": "user", "content": prompt}]
            )

            summary = response.choices[0].message.content.strip()
            logger.info("Successfully summarized parallel search results.")
            self.progress_callback("Consolidated findings from parallel search.")
            return summary
        except Exception as e:
            logger.error(f"Error summarizing results: {e}")
            logger.error(traceback.format_exc())
            # Fallback to returning the raw results on failure
            return "\n\n".join(parallel_results)

    def _get_available_tools(self) -> Dict[str, Dict]:
        """Get the list of available tools and their descriptions."""
        return {
            "search_by_embedding": {
                "description": "Search emails using semantic similarity. This is your most powerful tool.",
                "params": ["query", "top_k"],
                "param_descriptions": {
                    "query": "The search query text",
                    "top_k": "Number of results to return (default: 50)",
                },
            },
            "add_to_scratch_pad": {
                "description": "Add information to the scratch pad",
                "params": ["content"],
                "param_descriptions": {"content": "Text to add to the scratch pad"},
            },
            "get_scratch_pad": {
                "description": "Get the current content of the scratch pad",
                "params": [],
                "param_descriptions": {},
            },
            "respond": {
                "description": "Generate the final response to the user",
                "params": ["response_text"],
                "param_descriptions": {
                    "response_text": "The response to send to the user"
                },
            },
        }

    def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        """
        Execute a tool with the given parameters.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters

        Returns:
            Tool execution result
        """
        try:
            # Report tool execution to progress callback
            if self.progress_callback:
                tool_description = f"Using tool: {tool_name}"
                if tool_name == "search_by_embedding":
                    tool_description = f"Searching for emails similar to: '{params.get('query', '')[:50]}...'"
                elif tool_name == "add_to_scratch_pad":
                    tool_description = "Updating research notes..."

                self.progress_callback(tool_description)

            # Get the tool method
            tool_method = getattr(self.tools, tool_name)

            # Execute the tool
            result = tool_method(**params)

            # Log tool execution (full result)
            logger.info(f"Tool executed: {tool_name}")
            logger.info(f"Parameters: {json.dumps(params, indent=2)}")
            logger.info(
                f"Result: {str(result)[:500]}..."
            )  # Only truncate for console logging

            # Update scratch pad if needed
            if tool_name == "add_to_scratch_pad":
                logger.info(f"Current scratch pad:\n{self.tools.get_scratch_pad()}")

            # Return the full result
            return str(result)

        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            # Report error to progress callback
            if self.progress_callback:
                self.progress_callback(f"Error: {error_msg[:100]}...")

            return error_msg

    def research(self, query: str, existing_background: str = None) -> str:
        """
        Perform research on the user's query.

        Args:
            query: The user's research query
            existing_background: Optional existing background information to be updated.

        Returns:
            Final response to the user's query
        """
        logger.info(f"Starting research for query: {query}")

        # Reset state
        self.context = []
        self.tools.reset_scratch_pad()
        self.tools.start_research_logging(query)
        logger.info("Research state reset")

        if self.progress_callback:
            self.progress_callback(f"Starting research on: {query}")

        tools = self._get_available_tools()

        for i in range(self.max_iterations):
            self.progress_callback(f"Research iteration {i+1}/{self.max_iterations}")

            response, tool_call = self.model.call_model(
                query,
                tools,
                self.context,
                self.tools.get_scratch_pad(),
                i,
                self.max_iterations,
                research_tools=self.tools,
            )

            if not tool_call or tool_call.get("tool") == "respond":
                break

            result = self._execute_tool(tool_call["tool"], tool_call.get("params", {}))

            self.context.append(
                {
                    "tool": tool_call["tool"],
                    "params": tool_call.get("params", {}),
                    "result": result,
                }
            )

            self._extract_and_update_scratch_pad(
                query, tool_call["tool"], tool_call.get("params", {}), result
            )

        # After the loop, synthesize the final response from the scratchpad
        final_response = self._synthesize_final_response(query, existing_background)
        logger.info("Research completed")
        return final_response

    def _extract_and_update_scratch_pad(
        self, query: str, tool_name: str, params: Dict[str, Any], result: str
    ) -> None:
        """
        Process tool results to extract relevant information and update the scratch pad.

        Args:
            query: Original user query
            tool_name: Name of the tool that was executed
            params: Parameters used for the tool
            result: Result returned by the tool
        """
        logger.info(f"Extracting relevant information from {tool_name} result")

        extraction_system_prompt = """You are an information extraction specialist. Your task is to analyze research tool results and extract ONLY the most relevant information for answering a user's query.
Focus on concrete facts, not summaries or interpretations. Include specific details like dates, names, email addresses, and exact quotes when relevant.
Present information in a structured Markdown format with headers and bullet points. DO NOT include your own commentary, thoughts, or explanations.
Extract ONLY information that's relevant to the query. Include quotes from the emails when they're directly relevant to the query."""

        extraction_user_prompt = f"""Extract the most relevant information from the following tool result that will help answer the query.

USER QUERY: {query}
TOOL USED: {tool_name}
TOOL PARAMETERS: {json.dumps(params, indent=2)}

TOOL RESULT:
{result}

Extract ONLY the most relevant information that helps answer the query."""

        try:
            response = self.model.client.chat.completions.create(
                model=self.model.model,
                messages=[
                    {"role": "system", "content": extraction_system_prompt},
                    {"role": "user", "content": extraction_user_prompt},
                ],
            )

            extracted_info = response.choices[0].message.content.strip()

            if not extracted_info or extracted_info.lower().startswith(
                "no relevant information"
            ):
                logger.warning(
                    f"No relevant information extracted from {tool_name} result"
                )
                return

            self.tools.add_to_scratch_pad(extracted_info)
            self.tools.log_scratchpad()
            logger.info(
                f"Successfully updated scratch pad with information from {tool_name}"
            )

        except Exception as e:
            logger.error(
                f"Error extracting information from {tool_name} result: {str(e)}"
            )
            logger.error(traceback.format_exc())

    def _synthesize_final_response(
        self, query: str, existing_background: str = None
    ) -> str:
        """Synthesize the final response from the scratch pad content."""
        self.progress_callback("Synthesizing final response from research...")
        scratch_pad_content = self.tools.get_scratch_pad()

        if not scratch_pad_content.strip():
            if existing_background:
                self.progress_callback(
                    "No new information found. Keeping existing background."
                )
                return existing_background
            return "No information was found to answer the query."

        if existing_background:
            synthesis_prompt = f"""You are updating an existing user background profile with new analysis from recent emails. The user is the owner of the email account from which the information was extracted.

EXISTING BACKGROUND PROFILE:
{existing_background}

NEW ANALYSIS FROM RECENT EMAILS (Research Information):
{scratch_pad_content}

Please update the existing profile by:
1. Incorporating any new information that wasn't previously captured.
2. Updating sections where new information shows changes or developments.
3. Adding new contacts, projects, or work developments.
4. Preserving all existing valuable information.
5. Maintaining the same structure and professional tone.
6. Adding a note at the top indicating when this update was performed.
7. If no significant new information is found, indicate that in the update note but return the existing profile largely intact.

The goal is to have an enriched, current profile that builds upon the existing information rather than replacing it.

UPDATED AND MERGED BACKGROUND PROFILE:"""
        else:
            synthesis_prompt = f"""Based on the following research information, provide a detailed and comprehensive answer to the user's query. The query was written from the perspective of me, the email account's owner.
The response should be well-structured, in Markdown format, and directly address the user's question.
Synthesize the information from the research into a cohesive and detailed narrative.

USER QUERY: {query}

RESEARCH INFORMATION:
{scratch_pad_content}

Important: The research information may include inline citation markers of the form [[CID:<db_name>:<uid>]].
Instructions:
1. When you use information that has a [[CID:...]] marker, preserve that marker at the end of the sentence or bullet.
2. Do not invent new markers and do not modify the marker format.
3. Do not expand the marker; just keep it verbatim in the output.

DETAILED RESPONSE (preserve any [[CID:...]] markers present):"""

        try:
            response = self.model.client.chat.completions.create(
                model=self.model.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful research assistant providing final answers based on collected information.",
                    },
                    {"role": "user", "content": synthesis_prompt},
                ],
            )
            final_response = response.choices[0].message.content
            return final_response
        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            return "Failed to generate a response based on the research information."
