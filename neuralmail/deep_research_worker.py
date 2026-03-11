from PyQt5.QtCore import QThread, pyqtSignal
from .research_agent import ResearchAgent
import logging
import traceback
from typing import List, Dict
from .config import config
import os
import sqlite3
from pathlib import Path
from .utils import prepare_openrouter_request_params

logger = logging.getLogger(__name__)


class DeepResearchWorker(QThread):
    """Worker thread for performing deep research on a query."""

    progress = pyqtSignal(str)  # Signal for progress updates
    finished = pyqtSignal(str)  # Signal for final response
    error = pyqtSignal(str)  # Signal for errors

    def __init__(self, accounts: List[dict], query: str):
        """
        Initialize the deep research worker.

        Args:
            accounts: List of dictionaries containing account settings
            query: The research query to process
        """
        super().__init__()
        self.accounts = accounts
        self.query = query
        self.agents = []  # List to store research agents for each account
        self.timing_data = {
            "query_elaboration": 0,
            "research_processing": 0,
            "response_synthesis": 0,
            "total_research_time": 0,
        }
        logger.info(
            f"Deep research worker initialized for query: {query} across {len(accounts)} accounts"
        )

    def update_progress(self, message: str):
        """
        Update progress callback that will be passed to the research agents.

        Args:
            message: Progress message to report
        """
        # Emit the progress signal with the message
        self.progress.emit(message)
        logger.info(f"Research progress: {message}")

    def elaborate_query_with_background(self, query):
        """Lightly elaborate the research query using key background information."""
        try:
            import os

            if os.path.exists("background_info.txt"):
                with open("background_info.txt", "r", encoding="utf-8") as f:
                    background_info = f.read()
                    if background_info.strip():
                        # Use the global API key for elaboration
                        from research_agent import ResearchAgent

                        agent = ResearchAgent(
                            api_key=config.get("llm_api_key"),
                            embedding_api_key=config.get("embedding_api_key"),
                            db_paths=[self.accounts[0]["db_path"]],
                            progress_callback=self.update_progress,
                            base_url=config.get("llm_base_url"),
                            embedding_base_url=config.get("embedding_base_url"),
                            model=config.get("llm_model"),
                        )

                        elaboration_prompt = f"""You are an expert search assistant. Your task is to refine a user's research query by adding contextual keywords from their background information, without altering the original intent of the query.

**USER BACKGROUND:**
{background_info}

**ORIGINAL RESEARCH QUERY:** {query}

**Instructions:**
1.  **Preserve the Original Query**: The original query must be included in the refined query exactly as it is.
2.  **Add Contextual Keywords**: Append 1-3 relevant keywords from the user's background that will help focus the search, but only if it does not change the meaning.
3.  **Do Not Change the Intent**: The refined query must not change the core meaning or intent of the original query.
4.  **Output Only the Query**: Your final output should only be the refined search query string.

**Refined Search Query:**"""

                        # Prepare request parameters with OpenRouter provider routing
                        base_params = {
                            "model": config.get("llm_model"),
                            "messages": [{"role": "user", "content": elaboration_prompt}],
                            "max_tokens": 400,
                        }
                        request_params = prepare_openrouter_request_params(
                            config.get("llm_base_url"), config.get("llm_model"), base_params
                        )
                        
                        response = agent.openai_client.chat.completions.create(**request_params)

                        elaborated_query = response.choices[0].message.content.strip()
                        logger.info(f"Original research query: {query}")
                        logger.info(f"Elaborated research query: {elaborated_query}")
                        return elaborated_query
        except Exception as e:
            logger.error(f"Error elaborating research query: {e}")

        # Fallback to original query
        return query

    def run(self):
        """Run the research process."""
        import time
        import concurrent.futures

        total_start_time = time.time()

        try:
            # --- Query Elaboration ---
            elaboration_start_time = time.time()
            # No query elaboration is performed in this version, so we set the value to 0
            # In a future version, query elaboration logic can be added here
            self.timing_data["query_elaboration"] = 0
            
            # --- Research Processing ---
            research_start_time = time.time()

            # 1. Initialize a single research agent for all accounts
            self.progress.emit("Initializing research process...")
            db_paths = [
                account["db_path"]
                for account in self.accounts
                if account.get("db_path")
            ]
            if not db_paths:
                self.error.emit("No valid database paths found for any account.")
                return

            agent = ResearchAgent(
                api_key=config.get("llm_api_key"),
                embedding_api_key=config.get("embedding_api_key"),
                db_paths=db_paths,
                progress_callback=self.update_progress,
                base_url=config.get("llm_base_url"),
                embedding_base_url=config.get("embedding_base_url"),
                model=config.get("llm_model"),
            )
            agent.tools.start_research_logging(self.query)

            # 2. Initial Search (Single Query)
            self.progress.emit("Performing initial broad search...")
            initial_results = agent.tools.search_by_embedding(self.query, top_k=10)
            initial_summary = agent.summarize_results(
                self.query, [str(initial_results)]
            )
            agent.tools.add_to_scratch_pad(
                f"Initial findings for query '{self.query}':\n{initial_summary}"
            )

            # --- Parallel Research Loop (e.g., 2 rounds) ---
            num_parallel_rounds = 2
            for i in range(num_parallel_rounds):
                self.progress.emit(
                    f"--- Starting Parallel Research Round {i+1}/{num_parallel_rounds} ---"
                )

                # 3. Planning Step: Generate N parallel queries
                parallel_queries = agent.generate_parallel_queries(
                    self.query, agent.tools.get_scratch_pad(), num_queries=5
                )

                # 4. Parallel Execution Step
                self.progress.emit(
                    f"Executing {len(parallel_queries)} queries in parallel..."
                )
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Each thread will call the agent's search tool
                    future_to_query = {
                        executor.submit(agent.tools.search_by_embedding, q, top_k=10): q
                        for q in parallel_queries
                    }

                    parallel_results = []
                    for future in concurrent.futures.as_completed(future_to_query):
                        query_text = future_to_query[future]
                        try:
                            result = future.result()
                            parallel_results.append(str(result))
                            self.progress.emit(
                                f"Completed search for: '{query_text[:40]}...'"
                            )
                        except Exception as exc:
                            logger.error(
                                f"'{query_text}' generated an exception: {exc}"
                            )
                            self.progress.emit(
                                f"Error searching for: '{query_text[:40]}...'"
                            )

                # 5. Consolidation Step
                if parallel_results:
                    self.progress.emit(
                        "Consolidating results from parallel searches..."
                    )
                    summary = agent.summarize_results(self.query, parallel_results)
                    agent.tools.add_to_scratch_pad(
                        f"Summary from parallel round {i+1}:\n{summary}"
                    )
            
            self.timing_data["research_processing"] = time.time() - research_start_time

            # --- Final Synthesis ---
            synthesis_start_time = time.time()
            # 6. Final Synthesis with post-processing of citation IDs
            self.progress.emit("Synthesizing final response...")
            final_response = agent._synthesize_final_response(self.query)
            
            self.timing_data["response_synthesis"] = time.time() - synthesis_start_time

            # 7. Post-process: Replace [[CID:...]] with numeric citations and build source list
            try:
                cited_map = agent.tools.get_cited_emails()  # citation_id -> metadata
                if cited_map and isinstance(final_response, str):
                    import re

                    # Normalize grouped markers like [[CID:a], [CID:b]] to adjacent markers [[CID:a]][[CID:b]]
                    group_pattern = re.compile(
                        r"\[\[(?:CID:[^\]]+)\](?:,\s*\[(?:CID:[^\]]+)\])+\]"
                    )

                    def normalize_group(match):
                        inner = match.group(0)[2:-2]  # strip outer [[ ... ]]
                        parts = re.findall(r"\[(CID:[^\]]+)\]", inner)
                        return "".join([f"[[{p}]]" for p in parts])

                    final_response = group_pattern.sub(normalize_group, final_response)

                    # Find all unique CID markers in the final response
                    cid_pattern = re.compile(r"\[\[CID:([A-Za-z0-9_\-]+:[^\]]+)\]\]")
                    found_ids = cid_pattern.findall(final_response)
                    # Deduplicate preserving order
                    seen = set()
                    ordered_ids = []
                    for cid in found_ids:
                        if cid not in seen:
                            seen.add(cid)
                            ordered_ids.append(cid)

                    # Build mapping from cid -> index
                    cid_to_index = {cid: idx + 1 for idx, cid in enumerate(ordered_ids)}

                    # Replace markers in text
                    def replace_marker(match):
                        cid = match.group(1)
                        num = cid_to_index.get(cid)
                        return f"[{num}]" if num is not None else match.group(0)

                    final_response = cid_pattern.sub(replace_marker, final_response)

                    # Build source list
                    source_lines = ["\n\n---\n\n**Source Emails:**\n"]
                    for cid in ordered_ids:
                        meta = cited_map.get(cid, {})
                        # Fallback: if meta missing or core fields empty, try fetching directly from DB by cid
                        if not meta or not any(
                            meta.get(k)
                            for k in ("subject", "from_addr", "to_addr", "date")
                        ):
                            try:
                                db_name, uid = cid.split(":", 1)
                                # Prefer the DB with matching stem; fallback to scanning all
                                candidate_paths = [
                                    p
                                    for p in (locals().get("db_paths") or [])
                                    if Path(p).stem == db_name
                                ] or (locals().get("db_paths") or [])
                                for dbp in candidate_paths:
                                    try:
                                        conn = sqlite3.connect(dbp)
                                        c = conn.cursor()
                                        c.execute(
                                            """SELECT folder, subject, from_addr, to_addr, date FROM emails WHERE uid = ? LIMIT 1""",
                                            (uid,),
                                        )
                                        row = c.fetchone()
                                        conn.close()
                                        if row:
                                            meta = {
                                                "citation_id": cid,
                                                "db_path": dbp,
                                                "uid": uid,
                                                "folder": row[0] or "",
                                                "subject": row[1] or "",
                                                "from_addr": row[2] or "",
                                                "to_addr": row[3] or "",
                                                "date": row[4] or "",
                                            }
                                            break
                                    except Exception:
                                        try:
                                            conn.close()
                                        except Exception:
                                            pass
                            except Exception as fetch_err:
                                logger.error(
                                    f"Failed fallback metadata fetch for CID {cid}: {fetch_err}"
                                )
                        idx = cid_to_index[cid]
                        source_lines.append(
                            f"\n* **[{idx}]**\n"
                            f"  - **CID:** {cid}\n"
                            f"  - **Date:** {meta.get('date','')}\n"
                            f"  - **From:** {meta.get('from_addr','')}\n"
                            f"  - **To:** {meta.get('to_addr','')}\n"
                            f"  - **Subject:** {meta.get('subject','')}\n"
                        )

                    final_response += "".join(source_lines)
            except Exception as e:
                logger.error(f"Error during citation post-processing: {e}")

            self.timing_data["total_research_time"] = time.time() - total_start_time
            self.log_timing_summary()
            logger.info("Parallel research completed successfully.")
            self.finished.emit(final_response)

        except Exception as e:
            self.timing_data["total_research_time"] = time.time() - total_start_time
            self.log_timing_summary()
            error_msg = f"Error during parallel research: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.error.emit(error_msg)

    def log_timing_summary(self):
        """Log a summary of timing information for the deep research processing."""
        logger.info("=" * 60)
        logger.info("DEEP RESEARCH PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Query: {self.query}")
        logger.info("-" * 60)
        logger.info(
            f"Query Elaboration:            {self.timing_data['query_elaboration']:.3f}s"
        )
        logger.info(
            f"Research Processing:          {self.timing_data['research_processing']:.3f}s"
        )
        logger.info(
            f"Response Synthesis:           {self.timing_data['response_synthesis']:.3f}s"
        )
        logger.info("-" * 60)
        logger.info(
            f"TOTAL RESEARCH TIME:          {self.timing_data['total_research_time']:.3f}s"
        )
        logger.info("=" * 60)

    def stop(self):
        """Stop the research process."""
        logger.info("Stopping research process...")
        self.terminate()  # Force thread termination
