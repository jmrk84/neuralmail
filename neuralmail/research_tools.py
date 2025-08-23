import logging
import sqlite3
import numpy as np
import html2text
import email
from email.header import decode_header
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
import os
import json
from pathlib import Path
import time
import openai
import html
import traceback
from .utils import get_embedding, cosine_similarity
from .config import config

logger = logging.getLogger(__name__)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text for logging purposes."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


class ResearchTools:
    """Collection of tools for email research."""

    def __init__(
        self,
        db_paths: List[str],
        openai_client: Any,
        embedding_api_key: str = None,
        embedding_base_url: str = None,
    ):
        """
        Initialize research tools.

        Args:
            db_paths: List of paths to SQLite databases
            openai_client: OpenAI client instance
        """
        self.db_paths = db_paths
        self.openai_client = openai_client
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
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
            else self.openai_client
        )
        self._scratch_pad = ""
        self.current_query = None
        self.html_log = []
        self.log_dir = Path("research_logs")
        self.log_dir.mkdir(exist_ok=True)
        # Map of citation_id -> minimal email metadata used for citations
        # citation_id format: "<db_name>:<uid>"
        self.cited_emails: Dict[str, Dict[str, Any]] = {}
        logger.info(f"Initialized ResearchTools with databases: {db_paths}")

    def start_research_logging(self, query: str) -> None:
        """Start logging a new research query."""
        self.current_query = query
        self.html_log = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = self.log_dir / f"research_{timestamp}.html"
        self._log_html_section("Research Query", query, "query")

    def _log_html_section(
        self,
        title: str,
        content: Any,
        section_type: str = "default",
        collapsible: bool = False,
        expanded: bool = True,
        additional_content: Dict[str, Any] = None,
    ) -> None:
        """
        Add a section to the HTML log.

        Args:
            title: Section title
            content: Main content to log
            section_type: Type of section (affects styling)
            collapsible: Whether the section can be collapsed
            expanded: Whether the section is expanded by default
            additional_content: Additional content to include in the section
        """
        # Convert content to string if it's not already
        if content is None:
            content_str = "None"
        elif isinstance(content, (dict, list)):
            try:
                content_str = json.dumps(content, indent=2)
            except:
                content_str = str(content)
        else:
            content_str = str(content)

        # Note: Do not truncate content_str to ensure full logging

        # Generate a unique ID for the section
        section_id = f"section-{len(self.html_log)}"

        # Set section class based on type
        section_class = {
            "default": "log-section",
            "tool": "tool-section",
            "error": "error-section",
            "success": "success-section",
            "warning": "warning-section",
            "info": "info-section",
            "llm": "llm-section",
        }.get(section_type, "log-section")

        # Create the section header
        if collapsible:
            expanded_attr = "true" if expanded else "false"
            header = f"""
            <div class="{section_class} collapsible">
                <div class="section-header" onclick="toggleSection('{section_id}')" 
                     aria-expanded="{expanded_attr}" aria-controls="{section_id}">
                    <span class="toggle-icon">{"▼" if expanded else "►"}</span>
                    <h3>{html.escape(title)}</h3>
                </div>
                <div id="{section_id}" class="section-content {'expanded' if expanded else 'collapsed'}">
            """
        else:
            header = f"""
            <div class="{section_class}">
                <div class="section-header">
                    <h3>{html.escape(title)}</h3>
                </div>
                <div id="{section_id}" class="section-content">
            """

        # Create the content div with pre-formatted content
        content_div = f"""
            <pre class="content-pre">{html.escape(content_str)}</pre>
        """

        # Add additional content if provided
        additional_divs = ""
        if additional_content:
            for subtitle, subcontent in additional_content.items():
                subcontent_str = str(subcontent)
                additional_divs += f"""
                <div class="additional-content">
                    <h4>{html.escape(subtitle)}</h4>
                    <pre>{html.escape(subcontent_str)}</pre>
                </div>
                """

        # Close the section
        footer = """
                </div>
            </div>
        """

        # Combine all parts
        section_html = header + content_div + additional_divs + footer

        # Add to the log
        self.html_log.append(section_html)

        # Save log periodically
        if len(self.html_log) % 5 == 0:
            self._save_html_log()

    def log_scratchpad(self) -> None:
        """Log the current state of the scratch pad."""
        if self._scratch_pad:
            self._log_html_section(
                "Current Scratch Pad",
                self._scratch_pad,
                "info",
                collapsible=True,
                expanded=True,
            )
        else:
            self._log_html_section(
                "Current Scratch Pad",
                "(Empty)",
                "info",
                collapsible=True,
                expanded=True,
            )

    def log_llm_call(
        self,
        system_prompt: str,
        user_message: str,
        response_content: str,
        tool_call: Optional[Dict] = None,
    ) -> None:
        """
        Log an LLM call to the HTML log.

        Args:
            system_prompt: The system prompt used
            user_message: The user message sent to the LLM
            response_content: The LLM's response
            tool_call: The extracted tool call (if any)
        """
        # Log the entire LLM interaction without truncation
        self._log_html_section(
            "LLM Interaction",
            "See details below",
            "llm",
            collapsible=True,
            expanded=False,
            additional_content={
                "System Prompt": system_prompt,
                "User Message": user_message,
                "Response": response_content,
                "Extracted Tool Call": (
                    json.dumps(tool_call, indent=2) if tool_call else "None"
                ),
            },
        )

    def _save_html_log(self) -> None:
        """Save the current HTML log to file."""
        if not self.current_query or not self.html_log:
            return

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Research Log: {self.current_query}</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 20px auto;
                    padding: 0 20px;
                }}
                h1 {{
                    color: #333;
                    border-bottom: 2px solid #ccc;
                    padding-bottom: 10px;
                }}
                .timestamp {{
                    color: #666;
                    font-size: 0.9em;
                    margin-bottom: 20px;
                }}
                pre {{
                    background-color: #f8f8f8;
                    padding: 10px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
            </style>
            <script>
                function toggleSection(id) {{
                    const section = document.getElementById(id);
                    const button = document.getElementById('btn_' + id);
                    if (section.style.display === 'none') {{
                        section.style.display = 'block';
                        button.textContent = '▼';
                    }} else {{
                        section.style.display = 'none';
                        button.textContent = '►';
                    }}
                }}
            </script>
        </head>
        <body>
            <h1>Research Log</h1>
            <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            {"".join(self.html_log)}
        </body>
        </html>
        """

        with open(self.current_log_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Updated research log: {self.current_log_file}")

    def _execute_tool_with_logging(
        self, tool_name: str, params: Dict[str, Any], func: callable
    ) -> Any:
        """
        Execute a tool function with logging.

        Args:
            tool_name: Name of the tool being executed
            params: Tool parameters
            func: Function to execute

        Returns:
            Tool execution result
        """
        start_time = time.time()

        try:
            # Log tool execution start
            self._log_html_section(
                f"Executing Tool: {tool_name}",
                f"Parameters: {json.dumps(params, indent=2)}",
                "tool",
            )

            # Execute the function - this now calls the lambda with **params which is accepted by our fixed lambdas
            result = func(**params)

            # Log execution time
            execution_time = time.time() - start_time

            # Format result for display
            if isinstance(result, (list, dict)):
                result_str = json.dumps(result, indent=2)
            else:
                result_str = str(result)

            # Log successful execution
            self._log_html_section(
                f"Tool Result: {tool_name}",
                result_str,
                "success",
                collapsible=True,
                expanded=True,
                additional_content={"Execution Time": f"{execution_time:.2f} seconds"},
            )

            return result

        except Exception as e:
            # Log error
            error_message = f"Error executing {tool_name}: {str(e)}"
            stack_trace = traceback.format_exc()

            self._log_html_section(
                f"Tool Error: {tool_name}",
                error_message,
                "error",
                additional_content={"Stack Trace": stack_trace},
            )

            # Re-raise or return error message
            logger.error(error_message)
            logger.error(stack_trace)
            return f"Error executing tool {tool_name}: {str(e)}"

    def get_tool_definitions(self) -> Dict[str, Dict]:
        """Get definitions of all available tools."""
        return {
            "search_by_embedding": {
                "description": "Search emails using semantic similarity to the query",
                "params": ["query", "top_k"],
                "param_descriptions": {
                    "query": "Text to search for",
                    "top_k": "Number of results to return (default: 50)",
                },
            },
            "search_by_sender": {
                "description": "Find emails from a particular sender",
                "params": ["sender_email", "top_k"],
                "param_descriptions": {
                    "sender_email": "Email address of the sender",
                    "top_k": "Number of results to return (default: 10)",
                },
            },
            "search_by_recipient": {
                "description": "Find emails to a particular recipient",
                "params": ["recipient_email", "top_k"],
                "param_descriptions": {
                    "recipient_email": "Email address of the recipient",
                    "top_k": "Number of results to return (default: 10)",
                },
            },
            "search_by_date_range": {
                "description": "Find emails within a date range",
                "params": ["start_date", "end_date", "top_k"],
                "param_descriptions": {
                    "start_date": "Start date in YYYY-MM-DD format",
                    "end_date": "End date in YYYY-MM-DD format",
                    "top_k": "Number of results to return (default: 10)",
                },
            },
            "get_email_thread": {
                "description": "Retrieve all emails in the same conversation thread",
                "params": ["email_id"],
                "param_descriptions": {
                    "email_id": "ID of the email to find thread for"
                },
            },
            "add_to_scratch_pad": {
                "description": "Add information to the scratch pad",
                "params": ["content"],
                "param_descriptions": {"content": "Text to add to the scratch pad"},
            },
            "get_scratch_pad": {
                "description": "Retrieve the current content of the scratch pad",
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
            "list_email_addresses": {
                "description": "List all unique email addresses found in the database",
                "params": ["limit"],
                "param_descriptions": {
                    "limit": "Maximum number of email addresses to return"
                },
            },
        }

    def _connect_db(self, db_path: str) -> sqlite3.Connection:
        """Create a database connection."""
        return sqlite3.connect(db_path)

    def search_by_embedding(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Search emails using embedding similarity.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results
        """
        return self._execute_tool_with_logging(
            "search_by_embedding",
            {"query": query, "top_k": top_k},
            lambda **kwargs: self._search_by_embedding_impl(query, top_k),
        )

    def _search_by_date_range_impl(
        self,
        start_date: str,
        end_date: str,
        top_k: int = 10,
        folders: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Implementation of date range search across all databases."""
        all_results = []
        for db_path in self.db_paths:
            conn = self._connect_db(db_path)
            c = conn.cursor()

            try:
                # Build query
                params = [start_date, end_date]

                # Add folder conditions if specified
                folder_clause = ""
                if folders and len(folders) > 0:
                    folder_clause = (
                        "AND (" + " OR ".join(["folder = ?" for _ in folders]) + ")"
                    )
                    params.extend(folders)

                # Add limit
                params.append(top_k)

                # Execute query
                c.execute(
                    f"""SELECT id, uid, folder, subject, from_addr, to_addr, date, body, attachment_names, has_attachments
                        FROM emails 
                        WHERE date >= ? AND date <= ? {folder_clause}
                        ORDER BY date DESC
                        LIMIT ?""",
                    params,
                )

                results = [
                    self._format_email_result(row, db_path) for row in c.fetchall()
                ]
                all_results.extend(results)
            finally:
                conn.close()

        # Sort all results by date and return top_k
        all_results.sort(key=lambda x: x["date"], reverse=True)
        return all_results[:top_k]

    def search_by_date_range(
        self,
        start_date: str,
        end_date: str,
        top_k: int = 10,
        folders: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find emails within a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            top_k: Number of results to return
            folders: Optional list of folders to search in

        Returns:
            List of emails within the date range
        """
        return self._execute_tool_with_logging(
            "search_by_date_range",
            {
                "start_date": start_date,
                "end_date": end_date,
                "top_k": top_k,
                "folders": folders,
            },
            lambda **kwargs: self._search_by_date_range_impl(
                start_date, end_date, top_k, folders
            ),
        )

    def _add_to_scratch_pad_impl(self, content: str) -> str:
        """
        Add content to the scratch pad.

        Args:
            content: Content to add

        Returns:
            Updated scratch pad
        """
        if self._scratch_pad:
            self._scratch_pad += f"\n\n{content}"
        else:
            self._scratch_pad = content

        # Log the updated scratch pad
        self.log_scratchpad()

        return self._scratch_pad

    def add_to_scratch_pad(self, content: str) -> str:
        """Add information to the scratch pad."""
        return self._execute_tool_with_logging(
            "add_to_scratch_pad", {"content": content}, self._add_to_scratch_pad_impl
        )

    def _respond_impl(self, response_text: str) -> str:
        """Implementation of generating the final response."""
        # We could do additional processing here if needed in the future
        return response_text

    def respond(self, response_text: str) -> str:
        """Generate the final response to the user."""
        return self._execute_tool_with_logging(
            "respond", {"response_text": response_text}, self._respond_impl
        )

    def _search_by_sender_impl(
        self, sender_email: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Implementation of sender search across all databases."""
        all_results = []
        for db_path in self.db_paths:
            conn = self._connect_db(db_path)
            c = conn.cursor()

            try:
                c.execute(
                    """SELECT id, uid, folder, subject, from_addr, to_addr, date, body, attachment_names, has_attachments
                       FROM emails 
                       WHERE from_addr LIKE ?
                       LIMIT ?""",
                    (f"%{sender_email}%", top_k),
                )
                results = [
                    self._format_email_result(row, db_path) for row in c.fetchall()
                ]
                all_results.extend(results)
            finally:
                conn.close()

        return all_results[:top_k]

    def search_by_sender(
        self, sender_email: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find emails from a specific sender.

        Args:
            sender_email: Email address of the sender
            top_k: Number of results to return

        Returns:
            List of emails from the sender
        """
        return self._execute_tool_with_logging(
            "search_by_sender",
            {"sender_email": sender_email, "top_k": top_k},
            lambda **kwargs: self._search_by_sender_impl(sender_email, top_k),
        )

    def _search_by_recipient_impl(
        self, recipient_email: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Implementation of recipient search across all databases."""
        all_results = []
        for db_path in self.db_paths:
            conn = self._connect_db(db_path)
            c = conn.cursor()

            try:
                c.execute(
                    """SELECT id, uid, folder, subject, from_addr, to_addr, date, body, attachment_names, has_attachments
                       FROM emails 
                       WHERE to_addr LIKE ?
                       LIMIT ?""",
                    (f"%{recipient_email}%", top_k),
                )
                results = [
                    self._format_email_result(row, db_path) for row in c.fetchall()
                ]
                all_results.extend(results)
            finally:
                conn.close()

        return all_results[:top_k]

    def search_by_recipient(
        self, recipient_email: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find emails to a specific recipient.

        Args:
            recipient_email: Email address of the recipient
            top_k: Number of results to return

        Returns:
            List of emails to the recipient
        """
        return self._execute_tool_with_logging(
            "search_by_recipient",
            {"recipient_email": recipient_email, "top_k": top_k},
            lambda **kwargs: self._search_by_recipient_impl(recipient_email, top_k),
        )

    def _get_email_thread_impl(self, email_id: str) -> List[Dict[str, Any]]:
        """Implementation of thread retrieval across all databases."""
        try:
            db_name, numeric_email_id_str = email_id.split("-")
            numeric_email_id = int(numeric_email_id_str)
        except ValueError:
            logger.error(f"Invalid email_id format: {email_id}")
            return []

        source_db_path = next(
            (path for path in self.db_paths if Path(path).stem == db_name), None
        )
        if not source_db_path:
            logger.error(f"Could not find database for email {email_id}")
            return []

        base_subject = None
        conn = self._connect_db(source_db_path)
        c = conn.cursor()
        try:
            c.execute("SELECT subject FROM emails WHERE id = ?", (numeric_email_id,))
            result = c.fetchone()
            if result:
                subject = result[0]
                base_subject = re.sub(
                    r"^(?:Re|Fwd|Fw|FWD|RE|FW):\s*", "", subject, flags=re.IGNORECASE
                )
        finally:
            conn.close()

        if not base_subject:
            return []

        all_results = []
        for db_path in self.db_paths:
            conn = self._connect_db(db_path)
            c = conn.cursor()
            try:
                c.execute(
                    """SELECT id, uid, folder, subject, from_addr, to_addr, date, body, attachment_names, has_attachments
                       FROM emails 
                       WHERE subject LIKE ? OR subject LIKE ?
                       ORDER BY date""",
                    (f"%{base_subject}%", f"%Re: {base_subject}%"),
                )
                results = [
                    self._format_email_result(row, db_path) for row in c.fetchall()
                ]
                all_results.extend(results)
            finally:
                conn.close()

        all_results.sort(key=lambda x: x["date"])
        return all_results

    def get_email_thread(self, email_id: str) -> List[Dict[str, Any]]:
        """
        Get all emails in the same conversation thread.

        Args:
            email_id: ID of the email to find thread for

        Returns:
            List of emails in the thread
        """
        return self._execute_tool_with_logging(
            "get_email_thread",
            {"email_id": email_id},
            lambda **kwargs: self._get_email_thread_impl(email_id),
        )

    def get_scratch_pad(self) -> str:
        """Get the current content of the scratch pad."""
        return self._scratch_pad

    def reset_scratch_pad(self) -> None:
        """Reset the scratch pad to empty."""
        self._scratch_pad = ""
        logger.info("Reset scratch pad")

    def get_cited_emails(self) -> Dict[str, Dict[str, Any]]:
        """Return the mapping of citation_id to email metadata collected during searches."""
        return self.cited_emails

    def _list_email_addresses_impl(self, limit: int = 100) -> Dict[str, List[str]]:
        """Implementation of listing all unique email addresses across all databases."""
        all_senders = set()
        all_recipients = set()

        for db_path in self.db_paths:
            conn = self._connect_db(db_path)
            c = conn.cursor()

            try:
                # Get unique sender addresses
                c.execute(
                    """
                    SELECT DISTINCT from_addr FROM emails 
                    WHERE from_addr IS NOT NULL AND from_addr != ""
                """
                )

                for row in c.fetchall():
                    if row[0] and "@" in row[0]:
                        matches = re.findall(r"[\w\.-]+@[\w\.-]+", row[0])
                        all_senders.update(matches)

                # Get unique recipient addresses
                c.execute(
                    """
                    SELECT DISTINCT to_addr FROM emails 
                    WHERE to_addr IS NOT NULL AND to_addr != ""
                """
                )

                for row in c.fetchall():
                    if row[0] and "@" in row[0]:
                        matches = re.findall(r"[\w\.-]+@[\w\.-]+", row[0])
                        all_recipients.update(matches)
            except Exception as e:
                logger.error(f"Error listing email addresses from {db_path}: {e}")
            finally:
                conn.close()

        # Combine, sort, and limit results
        senders = sorted(list(all_senders))[:limit]
        recipients = sorted(list(all_recipients))[:limit]
        all_emails = sorted(list(all_senders.union(all_recipients)))

        return {"senders": senders, "recipients": recipients, "all": all_emails}

    def list_email_addresses(self, limit: int = 100) -> Dict[str, List[str]]:
        """
        List all unique email addresses found in the database.

        Args:
            limit: Maximum number of email addresses to return

        Returns:
            Dictionary of senders and recipients
        """
        return self._execute_tool_with_logging(
            "list_email_addresses",
            {"limit": limit},
            lambda **kwargs: self._list_email_addresses_impl(limit),
        )

    def _format_email_result(self, row, db_path: str) -> Dict[str, Any]:
        """
        Format a database row into a structured email result.

        Args:
            row: Database row with email data
            db_path: Path to the database from which the row was fetched

        Returns:
            Dictionary with formatted email information
        """
        db_name = Path(db_path).stem
        try:
            # Database schema: id, uid, folder, subject, from_addr, to_addr, date, body, attachment_names, has_attachments
            return {
                "id": f"{db_name}-{row[0]}",
                "uid": str(row[1]) if row[1] is not None else "",
                "folder": row[2],
                "subject": row[3] or "",
                "from_addr": row[4] or "",
                "to_addr": row[5] or "",
                "date": row[6] or "",
                "body": row[7] or "",
                "attachment_names": row[8] or "",
                "has_attachments": bool(row[9]) if len(row) > 9 else False,
                "db_path": db_path,
                # Include a stable citation id for downstream references
                "citation_id": f"{db_name}:{str(row[1]) if row[1] is not None else ''}",
            }
        except (IndexError, TypeError) as e:
            logger.error(f"Error formatting email result: {e}")
            return {
                "id": None,
                "uid": "",
                "folder": "",
                "subject": "",
                "from_addr": "",
                "to_addr": "",
                "date": "",
                "body": "",
                "attachment_names": "",
                "has_attachments": False,
                "db_path": db_path,
                "citation_id": f"{db_name}:",
            }

    def _search_by_embedding_impl(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Implementation of embedding-based search across all databases.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of formatted email results
        """
        try:
            # Generate embedding for the query
            query_embedding = get_embedding(query, self.embedding_client)
            if query_embedding is None:
                logger.error("Failed to generate embedding for query")
                return []

            all_similarities = []
            for db_path in self.db_paths:
                conn = self._connect_db(db_path)
                c = conn.cursor()

                try:
                    # Get all emails with embeddings
                    c.execute(
                        "SELECT id, uid, folder, subject, from_addr, to_addr, date, body, attachment_names, has_attachments, embedding FROM emails WHERE embedding IS NOT NULL"
                    )
                    emails = c.fetchall()

                    if not emails:
                        logger.warning(f"No emails with embeddings found in {db_path}")
                        continue

                    # Calculate similarities
                    for email in emails:
                        try:
                            email_embedding = np.frombuffer(email[10], dtype=np.float64)
                            similarity = cosine_similarity(
                                query_embedding, email_embedding
                            )
                            all_similarities.append((similarity, email[:10], db_path))
                        except Exception as e:
                            logger.error(
                                f"Error calculating similarity for email {email[0]} in {db_path}: {e}"
                            )
                            continue
                finally:
                    conn.close()

            # Sort by similarity and take top_k from the combined list
            all_similarities.sort(key=lambda x: x[0], reverse=True)
            top_emails = all_similarities[:top_k]

            # Format results
            results = []
            for similarity, email_data, db_path in top_emails:
                email_result = self._format_email_result(email_data, db_path)
                email_result["similarity"] = float(similarity)
                results.append(email_result)

                # Collect minimal citation metadata for later source listing
                cid = email_result.get("citation_id")
                if cid:
                    # Store only essential fields
                    self.cited_emails[cid] = {
                        "citation_id": cid,
                        "db_path": email_result.get("db_path"),
                        "uid": email_result.get("uid"),
                        "folder": email_result.get("folder"),
                        "subject": email_result.get("subject"),
                        "from_addr": email_result.get("from_addr"),
                        "to_addr": email_result.get("to_addr"),
                        "date": email_result.get("date"),
                    }

            return results

        except Exception as e:
            logger.error(f"Error in embedding search: {e}")
            return []
