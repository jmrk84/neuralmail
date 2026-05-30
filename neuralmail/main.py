# Set Windows App ID as the very first thing
import sys

if sys.platform == "win32":
    import ctypes

    myappid = "NeuralMail.EmailClient.1.0"
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
        pass

import os
import re
import html2text
from PyQt5.QtGui import QIcon
import sqlite3
import email
from email.header import decode_header
import numpy as np
import time
import logging
import random
import traceback
import imaplib
from typing import Dict
from .config import config, save_config


from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QMessageBox,
    QFileDialog,
    QProgressBar,
    QCheckBox,
    QGroupBox,
    QPlainTextEdit,
    QSpinBox,
    QSizePolicy,
    QComboBox,
    QRadioButton,
    QTextBrowser,
    QMenu,
    QTabWidget,
    QScrollArea,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QSize

import openai
import PyPDF2
import io
import tiktoken

import markdown2
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

from .parallel_email_sync import ParallelEmailSyncWorker
from .deep_research_worker import DeepResearchWorker
from .research_agent import ResearchAgent
from .utils import (
    decode_email_header,
    connect_imap,
    init_db,
    get_embedding,
    openai_api_client,
    cosine_similarity,
    truncate_messages_for_context,
    count_tokens,
    prepare_openrouter_request_params,
)

class BackgroundAnalysisWorker(QThread):
    """Worker thread for analyzing emails to create background information about the user."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, accounts):
        super().__init__()
        self.accounts = accounts

    def run(self):
        try:
            self.progress.emit("Starting background analysis...")

            existing_background = ""
            if os.path.exists("background_info.txt"):
                try:
                    with open("background_info.txt", "r", encoding="utf-8") as f:
                        existing_background = f.read()
                    if existing_background.strip():
                        self.progress.emit(
                            "Found existing background information - will update it"
                        )
                    else:
                        self.progress.emit(
                            "Existing background file is empty, creating new profile."
                        )
                except Exception as e:
                    logger.error(f"Error loading existing background: {e}")
                    self.progress.emit(
                        "Could not load existing background - creating new one"
                    )
            else:
                self.progress.emit(
                    "No existing background found - creating new profile"
                )

            if not config.get("llm_api_key"):
                self.error.emit("No valid OpenAI API key found")
                return

            db_paths = [
                account["db_path"]
                for account in self.accounts
                if account.get("db_path")
            ]
            if not db_paths:
                self.error.emit("No database paths found for any account.")
                return

            agent = ResearchAgent(
                api_key=config.get("llm_api_key"),
                embedding_api_key=config.get("embedding_api_key"),
                db_paths=db_paths,
                progress_callback=self.progress.emit,
                base_url=config.get("llm_base_url"),
                embedding_base_url=config.get("embedding_base_url"),
                model=config.get("llm_model"),
            )

            if existing_background.strip():
                # Formulate a query to find new information for an update
                initial_query = "Analyze recent emails to find any new projects, changes in roles, new key contacts, or shifts in communication patterns that would be relevant for updating my background profile."
                background_info = agent.research(
                    initial_query, existing_background=existing_background
                )
                action = "updated"
            else:
                # Create new background from scratch
                initial_query = """Gather comprehensive information about me to build a detailed background profile. The profile should be well-structured and cover the following areas:

- User Identity and Professional Role
- Key Contacts and Relationships
- Work Responsibilities and Expertise Areas
- Ongoing Projects and Initiatives
- Communication Patterns and Preferences
- Business Context and Industry Environment

Please perform a thorough analysis to create a comprehensive profile."""

                background_info = agent.research(initial_query)
                action = "created"

            if not background_info:
                self.error.emit("Background analysis failed to produce a result.")
                return

            with open("background_info.txt", "w", encoding="utf-8") as f:
                f.write(background_info)

            self.progress.emit(
                f"Background analysis completed and {action} in background_info.txt"
            )
            self.finished.emit(background_info)

        except Exception as e:
            logger.error(f"Error in background analysis: {e}")
            self.error.emit(str(e))


# Constants
MAX_SYNC_WORKERS = 10  # Maximum worker threads for parallel email sync

# Set up logging to a single file and console
# Configure file handler with UTF-8 encoding
file_handler = logging.FileHandler("neuralmail.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Configure console handler with UTF-8 encoding and error handling
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
# Handle encoding errors gracefully by replacing problematic characters
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass  # If reconfigure fails, continue with fallback

# Set formatter for both handlers
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
)
logger = logging.getLogger(__name__)


def prepare_openrouter_request_params(base_url: str, model: str, base_params: Dict) -> Dict:
    """
    Prepare request parameters with OpenRouter provider routing if specified.
    
    Args:
        base_url: The API base URL
        model: The model name (may contain provider after :)
        base_params: Base request parameters
        
    Returns:
        Updated request parameters with provider routing if applicable
    """
    request_params = base_params.copy()
    
    # Parse provider from model name if using OpenRouter
    if base_url and "openrouter.ai" in base_url and ":" in model:
        parts = model.split(":", 1)
        actual_model_name = parts[0]
        provider_name = parts[1]
        
        # Update model name to remove provider suffix
        request_params["model"] = actual_model_name
        
        # Add provider routing
        request_params["provider"] = {
            "only": [provider_name],
            "allow_fallbacks": False
        }
    
    return request_params


# Helper function to sanitize log messages with problematic Unicode characters
def sanitize_log_message(message):
    """
    Sanitize a log message to handle Unicode characters that might cause encoding issues.
    """
    if not isinstance(message, str):
        message = str(message)
    
    # Encode to ASCII with error handling - replaces problematic Unicode characters
    message = message.encode('ascii', errors='replace').decode('ascii')
    
    return message


# Helper function to write to log file only (not console)
def write_to_log_file_only(message):
    """Write a message to the log file without console output."""
    try:
        with open("neuralmail.log", "a", encoding="utf-8") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} INFO {message}\n")
            f.flush()
    except Exception as e:
        logger.error(f"Failed to write to log file: {e}")


def log_top_emails(emails_list, stage_name, query_type):
    """Log top emails metadata without full body content."""
    if not emails_list:
        write_to_log_file_only(f"{stage_name} - No emails found for {query_type}")
        return

    email_summary = f"{stage_name} - Top {len(emails_list)} emails for {query_type}:\n"
    for idx, (similarity, email_entry, account) in enumerate(emails_list, 1):
        # email_entry format: id, uid, folder, subject, from_addr, to_addr, date, body, ...
        body_preview = (
            " ".join(email_entry[7].split()[:15]) + "..."
            if email_entry[7]
            else "[No body]"
        )
        email_summary += f"  {idx}. Similarity: {similarity:.4f} | Account: {account['account_name']}\n"
        email_summary += f"     UID: {email_entry[1]} | Folder: {email_entry[2]}\n"
        email_summary += f"     From: {email_entry[4]} | To: {email_entry[5]}\n"
        email_summary += f"     Date: {email_entry[6]} | Subject: {email_entry[3]}\n"
        email_summary += f"     Body Preview: {body_preview}\n"
        email_summary += "     " + "-" * 50 + "\n"

    write_to_log_file_only(email_summary)


def get_folder_names(mail):
    """Retrieve folder names from the IMAP server."""
    result, folders = mail.list()
    folder_names = []
    if result == "OK":
        for folder in folders:
            folder_info = folder.decode()
            parts = folder_info.split(' "/" ')
            if len(parts) == 2:
                folder_name = parts[1].strip('"')
                folder_names.append(folder_name)
    return folder_names


def iter_block_items(parent):
    """
    Yields each paragraph and table child within *parent*, in document order.

    Each returned value is an instance of either Paragraph or Table.
    """
    for child in parent.element.body.iterchildren():
        if child.tag.endswith("p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("tbl"):
            yield Table(child, parent)


def extract_table_text(table, level=0):
    """
    Extracts text from a table, including handling nested tables.

    Parameters:
    - table (Table): The table object to extract text from.
    - level (int): The current nesting level (for indentation).

    Returns:
    - list of str: The extracted table texts.
    """
    table_text = []
    indent = "  " * level  # Indentation based on nesting level

    try:
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                cell_text = cell.text.strip().replace("\n", " ")
                # Check for nested tables within the cell
                nested_tables = cell.tables
                if nested_tables:
                    # Recursively extract nested table text
                    nested_text = []
                    for nested_table in nested_tables:
                        nested_text.extend(
                            extract_table_text(nested_table, level=level + 1)
                        )
                    cell_text += f"\n{indent}  Nested Table:"
                    cell_text += "\n".join(
                        [f"{indent}    {line}" for line in nested_text]
                    )
                row_data.append(cell_text)
            # Join cell texts with tabs for readability
            table_text.append(indent + "\t".join(row_data))
    except Exception as e:
        logger.error(f"Error extracting table at level {level}: {e}")

    return table_text


def extract_text_with_tables(file_stream, filename):
    """
    Extracts text from both paragraphs and tables in a .docx file from a BytesIO stream.

    Parameters:
    - file_stream (BytesIO): The BytesIO stream of the .docx file.
    - filename (str): The name of the file (for logging purposes).

    Returns:
    - str: The extracted text with table contents.
    """
    extracted_text = ""
    try:
        # Reset the stream position
        file_stream.seek(0)
        doc = Document(file_stream)
        logger.info(f"Opened attachment '{filename}' successfully.")

        # Iterate through all elements in the document in order
        for block in iter_block_items(doc):
            if isinstance(block, Paragraph):
                para_text = block.text.strip()
                if para_text:
                    extracted_text += para_text + "\n"
            elif isinstance(block, Table):
                logger.info(f"Processing a table in attachment '{filename}'...")
                table_text = extract_table_text(block)
                if table_text:
                    extracted_text += "Table:\n"
                    extracted_text += "\n".join(table_text) + "\n"

        logger.info(f"Successfully extracted text from {filename}")
    except Exception as e:
        logger.error(f"Error extracting text from '{filename}': {e}")
    return extracted_text


class QueryWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    streaming_chunk = pyqtSignal(str)  # New signal for streaming responses

    def __init__(self, accounts, query, email_cache):
        super().__init__()
        self.accounts = accounts
        self.query = query
        self.email_cache = email_cache
        self.timing_data = {
            "email_reranking": 0,
            "embedding_generation": 0,
            "cosine_similarity": 0,
            "attachment_fetching_imap": 0,
            "attachment_text_extraction": 0,
            "response_generation": 0,
            "total_query_time": 0,
            "prompt_tokens_reranking": 0,
            "completion_tokens_reranking": 0,
            "prompt_tokens_response_generation": 0,
            "completion_tokens_response_generation": 0,
        }
        self.augmented_query = ""

    def rerank_emails_with_llm(self, top_emails, original_query, client):
        """Re-rank emails using an LLM for better relevance."""
        if not top_emails:
            return []

        max_context = config.get("llm_max_context", 256000)

        emails_to_process = top_emails
        body_snippet_words = 50
        num_to_request = 40
        max_tokens_for_reranking = 200

        if max_context <= 64000:
            emails_to_process = top_emails[:100]
            body_snippet_words = 25
            num_to_request = 20
            max_tokens_for_reranking = 100  # Less tokens for smaller request
        email_summaries = ""
        for idx, (sim, email_entry, account) in enumerate(emails_to_process):
            subject = email_entry[3]
            from_addr = email_entry[4]
            to_addr = email_entry[5]
            date = email_entry[6]
            body_snippet = " ".join(email_entry[7].split()[:body_snippet_words])

            email_summaries += f"""[[{idx+1}]] (Similarity: {sim:.2f})
- From: {from_addr}
- To: {to_addr}
- Date: {date}
- Subject: {subject}
- Body Snippet: {body_snippet}...
---
"""

        background_context = ""
        try:
            if os.path.exists("background_info.txt"):
                with open("background_info.txt", "r", encoding="utf-8") as f:
                    background_info = f.read()
                    if background_info.strip():
                        background_context = f"""**User Background Information:**
{background_info[:1000]}{'...' if len(background_info) > 1000 else ''}

"""
        except Exception as e:
            logger.error(f"Error loading background info for re-ranking: {e}")

        current_date_time = time.strftime("%Y-%m-%d %H:%M:%S")

        rerank_prompt = f"""You are an intelligent email ranking assistant. Your task is to re-rank a list of emails based on their relevance to a user's query.
The current date and time is {current_date_time}. Use this information if the query is time-sensitive.

{background_context}**User Query:** "{original_query}"

**Emails to be re-ranked (with original similarity scores):**
{email_summaries}

**Instructions:**
1. Analyze the user's query and the provided email summaries.
2. Consider the user's background information for context.
3. Identify the emails that are most relevant to the query.
4. Your response must be a JSON object with a "rankings" array containing the numbers of the top {num_to_request} most relevant emails, in descending order of relevance.
5. For example: {{"rankings": [5, 12, 1, 35, 2]}}
6. Do not include any other text, explanation, or formatting. Only provide the JSON object.

**Top {num_to_request} re-ranked email numbers:**"""

        try:
            write_to_log_file_only(f"Re-ranking Input Prompt:\n{rerank_prompt}")

            # Check if we're using OpenAI's API or models that support response_format
            base_url = config.get("llm_base_url")
            model_name = config.get("llm_model")
            
            is_openai_api = (
                base_url is None or 
                "api.openai.com" in base_url
            )
            
            # Gemini models on OpenRouter also support response_format
            is_gemini_openrouter = (
                base_url and 
                "openrouter.ai" in base_url and 
                model_name and 
                "gemini" in model_name.lower()
            )
            
            supports_response_format = is_openai_api or is_gemini_openrouter
            
            # Parse provider from model name if using OpenRouter
            provider_name = None
            actual_model_name = model_name
            if base_url and "openrouter.ai" in base_url and ":" in model_name:
                parts = model_name.split(":", 1)
                actual_model_name = parts[0]
                provider_name = parts[1]
            
            # Debug output for provider detection
            print(f"DEBUG: Re-ranking provider detection:")
            print(f"   Base URL: {base_url}")
            print(f"   Model: {model_name}")
            print(f"   Actual model: {actual_model_name}")
            print(f"   Provider: {provider_name}")
            print(f"   OpenAI API: {is_openai_api}")
            print(f"   Gemini OpenRouter: {is_gemini_openrouter}")
            print(f"   Supports response_format: {supports_response_format}")
            
            # Prepare messages and truncate if necessary
            messages = [{"role": "user", "content": rerank_prompt}]
            truncated_messages = truncate_messages_for_context(messages, max_tokens_for_reranking)
            
            # Prepare request parameters
            request_params = {
                "model": actual_model_name,
                "messages": truncated_messages,
                "max_tokens": max_tokens_for_reranking,
                "temperature": 0.0,
            }
            
            # Only use response_format for APIs that support it
            if supports_response_format:
                request_params["response_format"] = {"type": "json_object"}
                print("DEBUG: Using JSON response format")
            else:
                print("DEBUG: Using plain text response format")
            
            # Add OpenRouter provider routing if specified
            if provider_name and base_url and "openrouter.ai" in base_url:
                request_params["provider"] = {
                    "only": [provider_name],
                    "allow_fallbacks": False
                }
                print(f"DEBUG: Using exclusive provider routing to: {provider_name}")

            response = client.chat.completions.create(**request_params)

            if response.usage:
                self.timing_data["prompt_tokens_reranking"] = response.usage.prompt_tokens
                self.timing_data["completion_tokens_reranking"] = (
                    response.usage.completion_tokens
                )

            reranked_indices_str = response.choices[0].message.content.strip()
            logger.info(f"LLM response for re-ranking: {reranked_indices_str}")

            # Parse the response - try JSON first, then fall back to comma-separated format
            reranked_indices = self._parse_reranking_response(reranked_indices_str)
            
            # If parsing fails, output debug info to console
            if not reranked_indices:
                print(f"\nDEBUG: Re-ranking failed!")
                print(f"Raw LLM response: '{reranked_indices_str}'")
                print(f"Response length: {len(reranked_indices_str)}")
                print(f"Response repr: {repr(reranked_indices_str)}")
                if reranked_indices_str:
                    print(f"First 200 chars: {reranked_indices_str[:200]}")
                else:
                    print("Response is empty!")

            reranked_emails = []
            seen_indices = set()
            for index in reranked_indices:
                if 0 <= index < len(top_emails) and index not in seen_indices:
                    reranked_emails.append(top_emails[index])
                    seen_indices.add(index)

            logger.info(f"Re-ranked {len(reranked_emails)} emails.")
            return reranked_emails[:num_to_request]

        except Exception as e:
            logger.error(f"Error re-ranking emails: {e}")
            logger.warning("Falling back to original similarity ranking.")
            return top_emails[:num_to_request]

    def _parse_reranking_response(self, response_str: str):
        """
        Parse the model's response to extract a list of email indices.
        Handles both JSON objects and plain text responses.
        """
        indices = []
        
        # First check if response is empty
        if not response_str or not response_str.strip():
            print("DEBUG: Empty response string received for re-ranking")
            return indices
        
        try:
            # First, try to parse as JSON
            import json
            data = json.loads(response_str)
            print(f"DEBUG: Successfully parsed JSON: {data}")
            
            if isinstance(data, dict) and "rankings" in data:
                # JSON object: {"rankings": [5, 12, 1, 35, 2]}
                rankings = data["rankings"]
                if isinstance(rankings, list):
                    indices = [int(i) - 1 for i in rankings if isinstance(i, (int, str)) and str(i).strip().isdigit()]
                    print(f"DEBUG: Extracted {len(indices)} indices from JSON object")
            elif isinstance(data, list):
                # Direct JSON list: [5, 12, 1, 35, 2]
                indices = [int(i) - 1 for i in data if isinstance(i, (int, str)) and str(i).strip().isdigit()]
                print(f"DEBUG: Extracted {len(indices)} indices from JSON array")
                        
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # JSON parsing failed, try to parse as comma-separated text
            print(f"DEBUG: JSON parsing failed: {e}")
            logger.info("JSON parsing failed, attempting to parse as comma-separated text")
            indices = self._parse_text_rankings(response_str)
        
        print(f"DEBUG: Final parsed indices: {indices}")
        return indices

    def _parse_text_rankings(self, text: str):
        """
        Parse rankings from plain text response.
        Handles various text formats that models might use.
        """
        indices = []
        
        print(f"DEBUG: Attempting to parse text: '{text}'")
        
        # Try comma-separated format first
        comma_parts = text.split(",")
        print(f"DEBUG: Split by comma: {comma_parts}")
        
        for i in comma_parts:
            i_clean = i.strip()
            if i_clean.isdigit():
                indices.append(int(i_clean) - 1)
        
        print(f"DEBUG: Found {len(indices)} comma-separated numbers")
        
        # If no comma-separated numbers found, try extracting all numbers
        if not indices:
            import re
            numbers = re.findall(r'\b\d+\b', text)
            print(f"DEBUG: Regex found numbers: {numbers}")
            indices = [int(n) - 1 for n in numbers if n.isdigit()]
            print(f"DEBUG: Converted to {len(indices)} indices")
        
        return indices

    def run(self):
        total_start_time = time.time()
        try:
            if not self.accounts:
                self.error.emit("No accounts configured")
                return

            if not config.get("embedding_api_key"):
                self.error.emit("Embedding API key is not configured in settings.")
                return

            # Create clients for LLM and embedding models
            llm_client = openai_api_client(
                config.get("llm_api_key"), config.get("llm_base_url")
            )
            embedding_client = openai_api_client(
                config.get("embedding_api_key"), config.get("embedding_base_url")
            )

            # STAGE 1: Initial Search
            logger.info("--- STAGE 1: Initial search ---")

            # 1.1 Generate embedding for the original query
            embedding_start_time = time.time()
            query_text_for_embedding = self.query.replace("\n", " ")
            query_embedding = get_embedding(query_text_for_embedding, embedding_client)
            self.timing_data["embedding_generation"] += (
                time.time() - embedding_start_time
            )

            # 1.2 Get top 200 emails from all accounts
            all_emails_stage1 = []
            for account in self.accounts:
                if (
                    not account["imap_host"]
                    or not account["imap_user"]
                    or not account["db_path"]
                ):
                    continue
                try:
                    db_path = account["db_path"]
                    cached_account_data = self.email_cache.get(db_path)
                    if not cached_account_data or not cached_account_data[0]:
                        logger.warning(
                            f"No cached emails found for {account['account_name']}. Skipping."
                        )
                        continue

                    # Fetch more emails per account for the first stage
                    account_emails = self.query_account_emails(
                        cached_account_data, query_embedding, account, top_n=100
                    )
                    if account_emails:
                        all_emails_stage1.extend(
                            [
                                (similarity, email_entry, account)
                                for similarity, email_entry in account_emails
                            ]
                        )
                except Exception as e:
                    logger.error(
                        f"Error querying account {account['account_name']} in Stage 1: {e}"
                    )

            all_emails_stage1.sort(key=lambda x: x[0], reverse=True)
            top_200_emails = all_emails_stage1[:200]

            # Log the top 20 emails from Stage 1 (initial query)
            log_top_emails(
                top_200_emails[:20], "STAGE 1", f"initial query: '{self.query}'"
            )

            if not top_200_emails:
                self.finished.emit("No relevant emails found across any accounts.")
                return

            # STAGE 2: Re-ranking with LLM
            logger.info("--- STAGE 2: Re-ranking with LLM ---")
            rerank_start_time = time.time()
            top_emails = self.rerank_emails_with_llm(
                top_200_emails, self.query, llm_client
            )
            self.timing_data["email_reranking"] = time.time() - rerank_start_time

            # Log the top 40 emails from Stage 2 (re-ranked)
            log_top_emails(
                top_emails, "STAGE 2", f"re-ranked for query: '{self.query}'"
            )

            # Generate response
            if top_emails:
                answer = self.generate_response(top_emails, self.query, llm_client)
                self.timing_data["total_query_time"] = time.time() - total_start_time
                self.log_timing_summary()
                self.finished.emit(answer)
            else:
                # Fallback to stage 1 results if re-ranking finds nothing
                logger.warning(
                    "Re-ranking with LLM yielded no results. Falling back to Stage 1 results."
                )
                top_emails_fallback = top_200_emails[:40]

                # Log the fallback emails being used
                log_top_emails(
                    top_emails_fallback,
                    "FALLBACK",
                    f"Stage 1 results for original query: '{self.query}'",
                )

                if top_emails_fallback:
                    answer = self.generate_response(
                        top_emails_fallback, self.query, llm_client
                    )
                    self.timing_data["total_query_time"] = (
                        time.time() - total_start_time
                    )
                    self.log_timing_summary()
                    self.finished.emit(answer)
                else:
                    self.timing_data["total_query_time"] = (
                        time.time() - total_start_time
                    )
                    self.log_timing_summary()
                    self.finished.emit(
                        "No relevant emails found across any accounts to answer the question."
                    )
        except Exception as e:
            logger.error(f"Error processing query across accounts: {e}")
            self.error.emit(str(e))

    def query_account_emails(
        self, cached_account_data, query_embedding, account, top_n=40
    ):
        email_entries, email_embeddings_matrix, email_norms = cached_account_data

        if not email_entries:
            return []

        similarity_start_time = time.time()

        # Unpack & Matrix Creation is done at cache population.
        # Email Norms calculation is done at cache population.

        # Batch compute cosine similarity
        query_norm = np.linalg.norm(query_embedding)

        # Handle cases where norms can be zero
        # Add a small epsilon to avoid division by zero
        denominator = query_norm * email_norms  # email_norms is from cache
        denominator[denominator == 0] = 1e-9  # Avoid division by zero

        dot_products = np.dot(email_embeddings_matrix, query_embedding)

        similarity_scores = dot_products / denominator

        # Combine emails with their similarity scores
        similarities = list(zip(similarity_scores, email_entries))

        # Sort and get top emails
        top_emails = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_n]

        self.timing_data["cosine_similarity"] += time.time() - similarity_start_time

        return top_emails

    def generate_response(self, top_emails, query, client):
        if not top_emails:
            return "No relevant emails found to answer the question."

        context = ""

        # Create a map to store connections by account
        account_connections = {}

        # This part now only times the IMAP connection and attachment processing loop
        attachment_processing_start = time.time()
        email_context_parts = []
        for idx, (sim, email_entry, account) in enumerate(top_emails):
            # Connect to the email server if needed for this account
            if account["imap_host"] not in account_connections:
                mail = self.connect_imap(
                    account["imap_host"],
                    account["imap_user"],
                    account["imap_pass"],
                    account["imap_port"],
                    account["use_ssl"],
                )
                if mail:
                    account_connections[account["imap_host"]] = mail
                else:
                    logger.error(
                        f"Failed to connect to the email server for {account['account_name']}"
                    )

            email_content = f"""[[{idx+1}]] (From account: {account['account_name']}):
From: {email_entry[4]}
To: {email_entry[5]}
Date: {email_entry[6]}
Subject: {email_entry[3]}

{email_entry[7]}
"""
            if email_entry[9]:  # has_attachments == 1
                mail = account_connections.get(account["imap_host"])
                if mail:
                    # This now internally updates the detailed attachment timing
                    attachment_text = self.process_attachments(
                        mail, email_entry, client, account
                    )
                    if attachment_text:
                        email_content += f"\nAttachment Text:\n{attachment_text}\n"

            email_context_parts.append(email_content)

        context = "\n" + "-" * 50 + "\n".join(email_context_parts)

        # Close all IMAP connections
        for mail in account_connections.values():
            mail.logout()
        # # After providing your answer, list the emails that you used to generate the answer, if any were found. The list should have one email per line, formatted as bullet points, including the **email number**, **account name**, **date**, **sender**, and **subject** information of each email.
        # Load background information if available
        background_context = ""
        try:
            if os.path.exists("background_info.txt"):
                with open("background_info.txt", "r", encoding="utf-8") as f:
                    background_info = f.read()
                    if background_info.strip():
                        background_context = f"""
## USER BACKGROUND INFORMATION
The following is background information about the email user to help provide context for your response:

{background_info}

---

"""
        except Exception as e:
            logger.error(f"Error loading background info for query: {e}")

        current_date_time = time.strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"""The current date and time is {current_date_time}. Use this information if the query is time-sensitive. The query was written from the perspective of me, the email account's owner.
   
   {background_context}Using the following emails, identified by `[[number]]`:
   
{context}

Please answer the user's question concisely and briefly (aim for about 200 tokens in your answer). Only provide more detailed responses if the user specifically asks for extensive detail, comprehensive analysis, or explicitly requests a longer response. 

Question: {query}

Your response must be nicely formatted in markdown. When citing sources, group them at the end of a sentence or paragraph preferentially to avoid long lists of numbers.

**Good Example:**
"The project was approved by Jane Doe and has a deadline of next Friday [4-10]." or "The project was approved by Jane Doe and has a deadline of next Friday [1]."

**Bad Example:**
"The project was approved by Jane Doe [1,2,3,4,5,6,7,8,9,10] and has a deadline of next Friday [5,6,7,8,9,10]."

NEVER use the word "Email" in your citations. Only use the bracketed numbers.
The email number corresponds to the list of source emails that will be appended to your response. Make sure to include citations for all key pieces of information.
"""
        try:
            gpt_start_time = time.time()

            # Prepare messages and truncate if necessary
            messages = [{"role": "user", "content": prompt}]
            truncated_messages = truncate_messages_for_context(messages, 4000)
            
            # Prepare request parameters with OpenRouter provider routing
            base_params = {
                "model": config.get("llm_model"),
                "messages": truncated_messages,
                "max_tokens": 4000,
                "stream": True,  # Enable streaming
                "stream_options": {
                    "include_usage": True
                },  # Request usage info with streaming
            }
            request_params = prepare_openrouter_request_params(
                config.get("llm_base_url"), config.get("llm_model"), base_params
            )
            
            # Create streaming response
            response_stream = client.chat.completions.create(**request_params)

            assistant_response = ""
            usage_info = None

            # Process streaming chunks
            for chunk in response_stream:
                # Check for usage information (may arrive in a chunk with empty choices)
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_info = chunk.usage

                if not chunk.choices:
                    continue

                # Check for content in the chunk
                if (
                    hasattr(chunk.choices[0].delta, "content")
                    and chunk.choices[0].delta.content
                ):
                    content = chunk.choices[0].delta.content
                    assistant_response += content
                    self.streaming_chunk.emit(content)

            self.timing_data["response_generation"] = time.time() - gpt_start_time

            # Capture usage information if available
            if usage_info:
                self.timing_data["prompt_tokens_response_generation"] = (
                    usage_info.prompt_tokens
                )
                self.timing_data["completion_tokens_response_generation"] = (
                    usage_info.completion_tokens
                )
            else:
                # Fallback: estimate tokens if usage info not available
                # Rough estimation: ~4 characters per token for English text
                estimated_input_tokens = len(prompt) // 4
                estimated_output_tokens = len(assistant_response) // 4
                self.timing_data["prompt_tokens_response_generation"] = (
                    estimated_input_tokens
                )
                self.timing_data["completion_tokens_response_generation"] = (
                    estimated_output_tokens
                )
                logger.warning(
                    "Usage information not available from streaming API, using token estimates"
                )

            # Append the list of source emails to the response
            email_list_str = "\n\n---\n\n**Source Emails:**\n"
            for idx, (sim, email_entry, account) in enumerate(top_emails):
                email_list_str += (
                    f"\n* **[{idx + 1}]** (Similarity: {sim:.2f})\n"
                    f"  - **Account:** {account['account_name']}\n"
                    f"  - **Date:** {email_entry[6]}\n"
                    f"  - **From:** {email_entry[4]}\n"
                    f"  - **To:** {email_entry[5]}\n"
                    f"  - **Subject:** {email_entry[3]}\n"
                )

            # Emit the source emails as a final chunk
            self.streaming_chunk.emit(email_list_str)

            return assistant_response + email_list_str
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            # Ensure timing is captured even on error
            if "gpt_start_time" in locals():
                self.timing_data["response_generation"] = time.time() - gpt_start_time
            return "An error occurred while generating the response."

    def log_timing_summary(self):
        """Log a summary of timing information for the query processing."""
        # Calculate sum of individual components
        component_sum = (
            self.timing_data["email_reranking"]
            + self.timing_data["embedding_generation"]
            + self.timing_data["cosine_similarity"]
            + self.timing_data["attachment_fetching_imap"]
            + self.timing_data["attachment_text_extraction"]
            + self.timing_data["response_generation"]
        )

        logger.info("=" * 60)
        logger.info("QUERY PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Query: {self.query}")
        logger.info("-" * 60)
        total_reranking_tokens = (
            self.timing_data["prompt_tokens_reranking"]
            + self.timing_data["completion_tokens_reranking"]
        )
        logger.info(
            f"Email Re-ranking (GPT):       {self.timing_data['email_reranking']:.3f}s (Prompt: {self.timing_data['prompt_tokens_reranking']}, Completion: {self.timing_data['completion_tokens_reranking']}, Total: {total_reranking_tokens} tokens)"
        )
        logger.info(
            f"Embedding Generation:         {self.timing_data['embedding_generation']:.3f}s"
        )
        logger.info(
            f"Cosine Similarity Compute:    {self.timing_data['cosine_similarity']:.3f}s"
        )
        logger.info(
            f"Attachment Fetching (IMAP):   {self.timing_data['attachment_fetching_imap']:.3f}s"
        )
        logger.info(
            f"Attachment Text Extraction:   {self.timing_data['attachment_text_extraction']:.3f}s"
        )
        logger.info(
            f"Response Generation (GPT):    {self.timing_data['response_generation']:.3f}s (Input: {self.timing_data['prompt_tokens_response_generation']} tokens, Output: {self.timing_data['completion_tokens_response_generation']} tokens)"
        )
        logger.info("-" * 60)
        logger.info(f"Sum of Components:            {component_sum:.3f}s")
        logger.info(
            f"ACTUAL TOTAL TIME:            {self.timing_data['total_query_time']:.3f}s"
        )
        if abs(component_sum - self.timing_data["total_query_time"]) > 0.1:
            # This difference highlights overhead not captured in the individual components
            logger.info(
                f"Unaccounted Time (overhead):  {self.timing_data['total_query_time'] - component_sum:.3f}s"
            )
        logger.info("=" * 60)

    def process_attachments(self, mail, email_entry, client, account):
        extracted_text = ""
        attachment_names = email_entry[8].split(",")
        folder = email_entry[2]  # Folder where the email is located

        # Select the folder/mailbox before fetching - ensure proper quoting
        folder_name = f'"{folder}"'  # Always quote the folder name
        result, data = mail.select(folder_name, readonly=True)
        if result != "OK":
            logger.error(
                f'Failed to select folder {folder_name} for attachments in {account["account_name"]}: {data}'
            )
            return extracted_text

        for attachment_name in attachment_names:
            if not attachment_name.lower().endswith((".pdf", ".doc", ".docx")):
                continue  # Skip non-supported file types

            # Time the IMAP fetch operation
            fetch_start_time = time.time()
            result, data = mail.uid("fetch", email_entry[1], "(RFC822)")
            self.timing_data["attachment_fetching_imap"] += (
                time.time() - fetch_start_time
            )

            if result != "OK":
                logger.error(
                    f'Failed to fetch email UID {email_entry[1]} for attachments in {account["account_name"]}: {data}'
                )
                continue

            email_message = email.message_from_bytes(data[0][1])
            for part in email_message.walk():
                content_disposition = str(part.get("Content-Disposition"))
                if "attachment" in content_disposition:
                    filename = part.get_filename()
                    if filename == attachment_name:
                        attachment_data = part.get_payload(decode=True)
                        logger.info(f"Processing attachment: {filename}")

                        # Time the text extraction process
                        extraction_start_time = time.time()

                        if attachment_name.lower().endswith(".pdf"):
                            try:
                                with io.BytesIO(attachment_data) as pdf_file:
                                    reader = PyPDF2.PdfReader(pdf_file)
                                    for page in reader.pages:
                                        page_text = page.extract_text()
                                        extracted_text += page_text
                                        if len(extracted_text) >= config.get(
                                            "max_tokens"
                                        ):
                                            break
                            except Exception as e:
                                logger.error(
                                    f"Error reading PDF attachment '{filename}' in {account['account_name']}: {e}"
                                )
                        elif attachment_name.lower().endswith((".doc", ".docx")):
                            try:
                                with io.BytesIO(attachment_data) as doc_file:
                                    # Extract text including tables
                                    extracted_text += extract_text_with_tables(
                                        doc_file, filename
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error reading Word attachment '{filename}' in {account['account_name']}: {e}"
                                )

                        self.timing_data["attachment_text_extraction"] += (
                            time.time() - extraction_start_time
                        )

        return extracted_text[: config.get("max_tokens")] if extracted_text else ""

    def connect_imap(
        self, imap_host, imap_user, imap_pass, imap_port=993, use_ssl=True
    ):
        try:
            if use_ssl:
                mail = imaplib.IMAP4_SSL(imap_host, imap_port)
            else:
                mail = imaplib.IMAP4(imap_host, imap_port)
            mail.login(imap_user, imap_pass)
            logger.info(
                "Successfully connected to IMAP server for attachment processing."
            )
            return mail
        except Exception as e:
            logger.error(f"Error connecting to IMAP server: {e}")
            return None


class MainWindow(QWidget):
    def __init__(self):
        try:
            logger.info("Starting application initialization...")
            super().__init__()
            logger.info("Base QWidget initialized")

            self.setWindowTitle("NeuralMail v1.1")
            self.settings = {}
            self.accounts = []  # List to store multiple account settings
            self.current_account_index = -1  # Index of currently selected account
            logger.info("Basic attributes initialized")

            # Set up DPI-aware icon handling
            self.setup_application_icon()
            logger.info("Window icon set")

            self.conn = None  # For main thread database operations
            self.folder_names = []
            self.emails_synced = False
            self.sync_workers = []
            self.sync_completed_count = 0
            self.sync_error_count = 0
            self.last_sync_error_message = ""
            self.total_sync_accounts = 0
            self.query_worker = None
            self.research_worker = None
            self.folder_checkboxes = []
            self.email_cache = {}
            self.streaming_buffer = ""
            self.streaming_started = False  # Flag to track if streaming has begun
            logger.info("Worker attributes initialized")

            screen = QApplication.primaryScreen()
            if screen:
                avail = screen.availableGeometry()
                w = min(900, avail.width() - 50)
                h = min(500, avail.height() - 50)
                self.resize(w, h)
            else:
                self.resize(900, 500)
            logger.info("Initial window size set")

            # Set up logging
            self.logger = logging.getLogger(__name__)
            logger.info("Logger initialized")

            # Initialize UI
            logger.info("Starting UI initialization...")
            self.init_ui()
            logger.info("UI initialized")
            
            # Set DPI-aware minimum size after UI is created
            self.set_dpi_aware_minimum_size()
            logger.info("DPI-aware minimum size set")

            # Load settings
            logger.info("Loading settings from config...")
            self.load_settings_from_config()
            logger.info("Settings loaded")

            # Populate initial cache
            self.populate_initial_cache()

            # Initialize threading variables
            self.query_thread = None
            logger.info("Threading variables initialized")

            # Initialize background information
            self.background_info = ""
            self.background_worker = None
            self.load_background_info()

            logger.info("Application initialization completed successfully")

            # Setup auto-sync timer only if at least one account is fully configured
            self.sync_timer = QTimer(self)
            self.sync_timer.timeout.connect(self.sync_emails)
            has_syncable_account = any(
                bool(account.get("imap_host"))
                and bool(account.get("imap_user"))
                and bool(account.get("imap_pass"))
                and bool(account.get("selected_folders"))
                for account in self.accounts
            )
            if has_syncable_account:
                self.sync_timer.start(300000)  # 5 minutes in milliseconds
                self.sync_emails()  # Initial sync on startup
            else:
                logger.info(
                    "No fully configured accounts found on startup; skipping initial sync and timer start."
                )

        except Exception as e:
            logger.error(f"Error during application initialization: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Show error message box
            QMessageBox.critical(
                None,
                "Initialization Error",
                f"Error initializing application:\n{str(e)}\n\nCheck neuralmail.log for details.",
            )
            raise  # Re-raise the exception to ensure the application closes properly

    def setup_application_icon(self):
        """Set up window icon - application icon is already set at app level."""
        try:
            icon_path = "icon.ico"
            if os.path.exists(icon_path):
                # Create icon with multiple sizes for better display
                icon = QIcon()
                icon.addFile(icon_path)  # Add base icon

                # Add specific sizes for better DPI support
                sizes = [16, 24, 32, 48, 64, 128, 256]
                for size in sizes:
                    icon.addFile(icon_path, QSize(size, size))

                # Only set the window icon (app icon is set at application level)
                self.setWindowIcon(icon)
                logger.info(f"Window icon set successfully from {icon_path}")
            else:
                logger.warning(f"Icon file not found: {icon_path}")

        except Exception as e:
            logger.error(f"Error setting up window icon: {e}")
            # Simple fallback
            try:
                if os.path.exists("icon.ico"):
                    self.setWindowIcon(QIcon("icon.ico"))
            except Exception as fallback_error:
                logger.error(f"Fallback icon loading also failed: {fallback_error}")

    def set_dpi_aware_minimum_size(self):
        """Set DPI-aware minimum size based on screen DPI."""
        try:
            screen = QApplication.primaryScreen()
            dpi = screen.logicalDotsPerInch()
            dpi_scale = dpi / 96.0

            base_min_width = 600
            base_min_height = 400
            scaled_min_width = int(base_min_width * dpi_scale)
            scaled_min_height = int(base_min_height * dpi_scale)

            screen_geom = screen.availableGeometry()
            scaled_min_width = min(scaled_min_width, screen_geom.width() - 50)
            scaled_min_height = min(scaled_min_height, screen_geom.height() - 50)

            self.setMinimumSize(scaled_min_width, scaled_min_height)
            logger.info(f"Set DPI-aware minimum size: {scaled_min_width}x{scaled_min_height} (DPI: {dpi:.1f}, scale: {dpi_scale:.2f})")

        except Exception as e:
            logger.error(f"Error setting DPI-aware minimum size: {e}")
            self.setMinimumSize(600, 400)

    def init_ui(self):
        self.setStyleSheet(
            """
            QWidget {
                background-color: #ffffff;
                color: #333333;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
            }
            QTabWidget::pane {
                border-top: 2px solid #f0f0f0;
            }
            QTabBar::tab {
                background: #f9f9f9;
                border: 1px solid #e0e0e0;
                border-bottom: none;
                padding: 8px 15px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border: 1px solid #d0d0d0;
                border-bottom: 1px solid #ffffff;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QLineEdit, QPlainTextEdit, QTextBrowser {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 5px;
                background-color: #fdfdfd;
            }
            QGroupBox {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """
        )

        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Create Search Tab
        self.search_tab = QWidget()
        self.query_layout = QVBoxLayout(self.search_tab)
        self.query_layout.setContentsMargins(20, 10, 20, 10)
        self.query_layout.setSpacing(10)
        self.init_query_tab()
        self.tabs.addTab(self.search_tab, "Search")

        # Create Settings Tab (scrollable)
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QScrollArea.NoFrame)
        self.settings_tab = QWidget()
        self.settings_layout = QVBoxLayout(self.settings_tab)
        self.settings_layout.setAlignment(Qt.AlignTop)
        self.init_settings_tab()
        settings_scroll.setWidget(self.settings_tab)
        self.tabs.addTab(settings_scroll, "Settings")

        # Persistent sync status line at the bottom of the window.
        # Replaces modal pop-ups for sync results (e.g. connection loss).
        self.sync_status_label = QLabel("Last sync: never")
        self.sync_status_label.setStyleSheet(
            "color: #666666; font-size: 9pt; padding: 2px 8px;"
        )
        main_layout.addWidget(self.sync_status_label)

        self.setLayout(main_layout)

    def init_settings_tab(self):
        """Initialize the settings tab components."""
        left_layout = QVBoxLayout()

        # Labels for the settings page
        title_label = QLabel("NeuralMail Settings")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(title_label)

        intro_label = QLabel("Configure your email accounts and AI settings")
        intro_label.setStyleSheet("margin-bottom: 15px;")
        left_layout.addWidget(intro_label)

        # Account selector and management
        account_group = QGroupBox("Account Management")
        account_layout = QVBoxLayout()

        account_selector_layout = QHBoxLayout()
        self.account_selector = QComboBox()
        self.account_selector.currentIndexChanged.connect(self.select_account)
        account_selector_layout.addWidget(QLabel("Selected Account:"))
        account_selector_layout.addWidget(self.account_selector, 1)

        account_buttons_layout = QHBoxLayout()
        self.add_account_button = QPushButton("Add Account")
        self.add_account_button.clicked.connect(self.add_account)
        account_buttons_layout.addWidget(self.add_account_button)

        self.delete_account_button = QPushButton("Delete Account")
        self.delete_account_button.clicked.connect(self.delete_account)
        self.delete_account_button.setEnabled(False)
        account_buttons_layout.addWidget(self.delete_account_button)

        self.background_button = QPushButton("Generate Background")
        self.background_button.clicked.connect(self.generate_background_info)
        account_buttons_layout.addWidget(self.background_button)

        account_layout.addLayout(account_selector_layout)
        account_layout.addLayout(account_buttons_layout)
        account_group.setLayout(account_layout)
        left_layout.addWidget(account_group)

        # LLM Provider Settings
        llm_group = QGroupBox("LLM Provider Settings")
        llm_layout = QFormLayout()
        self.llm_api_key = QLineEdit()
        self.llm_api_key.setEchoMode(QLineEdit.Password)
        self.llm_base_url = QLineEdit()
        self.llm_base_url.setPlaceholderText("Leave empty for OpenAI")
        self.llm_model = QLineEdit()
        self.llm_max_context = QSpinBox()
        self.llm_max_context.setRange(32000, 1000000)  # Up to 1M tokens
        self.llm_max_context.setValue(256000)  # Default to 256k
        llm_layout.addRow("API Key:", self.llm_api_key)
        llm_layout.addRow("Base URL:", self.llm_base_url)
        llm_layout.addRow("Model:", self.llm_model)
        llm_layout.addRow("Max Context (tokens):", self.llm_max_context)
        llm_group.setLayout(llm_layout)
        left_layout.addWidget(llm_group)

        # Embedding Provider Settings
        embedding_group = QGroupBox("Embedding Provider Settings")
        embedding_layout = QFormLayout()
        self.embedding_api_key = QLineEdit()
        self.embedding_api_key.setEchoMode(QLineEdit.Password)
        self.embedding_base_url = QLineEdit()
        self.embedding_base_url.setPlaceholderText("Leave empty for OpenAI")
        self.embedding_model = QLineEdit()
        embedding_layout.addRow("API Key:", self.embedding_api_key)
        embedding_layout.addRow("Base URL:", self.embedding_base_url)
        embedding_layout.addRow("Model:", self.embedding_model)
        embedding_group.setLayout(embedding_layout)
        left_layout.addWidget(embedding_group)

        # IMAP Settings
        imap_group = QGroupBox("IMAP Settings")
        imap_layout = QFormLayout()

        self.imap_host = QLineEdit()
        self.imap_user = QLineEdit()
        self.imap_pass = QLineEdit()
        self.imap_pass.setEchoMode(QLineEdit.Password)
        self.imap_port = QSpinBox()
        self.imap_port.setMaximum(65535)
        self.imap_port.setValue(993)
        self.use_ssl = QCheckBox("Use SSL")
        self.use_ssl.setChecked(True)
        imap_layout.addRow("Host:", self.imap_host)
        imap_layout.addRow("Username:", self.imap_user)
        imap_layout.addRow("Password:", self.imap_pass)
        imap_layout.addRow("Port:", self.imap_port)
        imap_layout.addRow(self.use_ssl)
        imap_group.setLayout(imap_layout)

        # Database Settings
        db_group = QGroupBox("Database Settings")
        db_layout = QHBoxLayout()

        self.db_path = QLineEdit("emails.db")
        self.db_browse = QPushButton("Browse")
        self.db_browse.clicked.connect(self.browse_db)
        db_layout.addWidget(self.db_path)
        db_layout.addWidget(self.db_browse)
        db_group.setLayout(db_layout)

        self.folder_group = QGroupBox("Select Folders")
        folder_group_layout = QVBoxLayout()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(100)
        scroll_area.setMaximumHeight(250)

        scroll_content = QWidget()
        self.folder_layout = QVBoxLayout(scroll_content)
        self.folder_layout.setAlignment(Qt.AlignTop)
        scroll_content.setLayout(self.folder_layout)

        scroll_area.setWidget(scroll_content)
        folder_group_layout.addWidget(scroll_area)
        self.folder_group.setLayout(folder_group_layout)

        self.folder_checkboxes = []

        buttons_layout = QHBoxLayout()
        self.sync_button = QPushButton("Sync Emails")
        self.sync_button.clicked.connect(self.sync_emails)
        buttons_layout.addWidget(self.sync_button)

        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        buttons_layout.addWidget(self.save_button)

        left_layout.addWidget(imap_group)
        left_layout.addWidget(db_group)
        left_layout.addWidget(self.folder_group)
        left_layout.addLayout(buttons_layout)
        left_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 1)
        self.settings_layout.addLayout(main_layout)

    def init_query_tab(self):
        """Initialize the query tab components."""
        query_layout = QVBoxLayout()
        query_layout.setContentsMargins(50, 20, 50, 20)

        # Query input
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask your emails :)")
        self.query_input.returnPressed.connect(self.search_emails)
        self.query_input.setFixedHeight(40)
        self.query_input.setStyleSheet(
            "font-size: 16px; border-radius: 20px; padding: 0 15px;"
        )
        query_layout.addWidget(self.query_input)

        # Deep research layout
        deep_research_layout = QHBoxLayout()

        self.deep_research_checkbox = QCheckBox("Deep Research (beta)")

        info_label = QLabel("ⓘ")
        info_label.setToolTip(
            "Deep Research performs a more thorough, intelligent analysis of your emails.\nThis may take longer (1 - 2 minutes) but can provide more comprehensive answers."
        )
        info_label.setStyleSheet("font-size: 12pt; color: #0078d4;")

        deep_research_layout.addStretch()
        deep_research_layout.addWidget(self.deep_research_checkbox)
        deep_research_layout.addWidget(info_label)
        deep_research_layout.addStretch()

        query_layout.addLayout(deep_research_layout)

        # Results area
        # QTextBrowser for HTML support
        self.results_text_edit = QTextBrowser()
        self.results_text_edit.setReadOnly(True)
        self.results_text_edit.setOpenExternalLinks(True)  # Enable clickable links
        self.results_text_edit.setMinimumHeight(100)
        self.results_text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_text_edit.setContextMenuPolicy(Qt.NoContextMenu)
        query_layout.addWidget(self.results_text_edit, 1)  # Give it stretch factor of 1

        # Add progress bar
        self.query_progress = QProgressBar()
        self.query_progress.setVisible(False)
        query_layout.addWidget(self.query_progress)

        self.query_layout.addLayout(query_layout)

    def browse_db(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Select Database File", "", "SQLite Database (*.db)"
        )
        if file_name:
            self.db_path.setText(file_name)

    def save_settings(self):
        """Save the current account settings."""
        # Update current account settings
        if self.current_account_index >= 0 and self.current_account_index < len(
            self.accounts
        ):
            current_account = self.accounts[self.current_account_index]
            current_account["imap_host"] = self.imap_host.text()
            current_account["imap_user"] = self.imap_user.text()
            current_account["imap_pass"] = self.imap_pass.text()
            current_account["imap_port"] = self.imap_port.value()
            current_account["use_ssl"] = self.use_ssl.isChecked()
            current_account["db_path"] = self.db_path.text()

            # Update folders
            selected_folders = self.get_selected_folders()
            current_account["selected_folders"] = selected_folders

            # Update account name in selector if username changed
            if current_account["imap_user"]:
                current_account["account_name"] = (
                    f"Account for {current_account['imap_user']}"
                )
                self.update_account_selector()
                self.account_selector.setCurrentIndex(self.current_account_index)

        # Update global settings in the config object
        config["llm_api_key"] = self.llm_api_key.text()
        config["llm_base_url"] = self.llm_base_url.text()
        config["llm_model"] = self.llm_model.text()
        config["llm_max_context"] = self.llm_max_context.value()
        config["embedding_api_key"] = self.embedding_api_key.text()
        config["embedding_base_url"] = self.embedding_base_url.text()
        config["embedding_model"] = self.embedding_model.text()
        config["accounts"] = self.accounts

        # Save all settings to the config file
        self.save_settings_to_config()

        QMessageBox.information(self, "Success", "Settings saved successfully.")
        logger.info("Settings saved.")

    def connect_imap(
        self, imap_host, imap_user, imap_pass, imap_port=993, use_ssl=True
    ):
        return connect_imap(imap_host, imap_user, imap_pass, imap_port, use_ssl)

    def load_settings_from_config(self):
        """Load settings from the config file."""
        self.accounts = config.get("accounts", [])

        # Load global AI settings first
        self.llm_api_key.setText(config.get("llm_api_key", ""))
        self.llm_base_url.setText(config.get("llm_base_url", ""))
        self.llm_model.setText(config.get("llm_model", ""))
        self.llm_max_context.setValue(config.get("llm_max_context", 256000))
        self.embedding_api_key.setText(config.get("embedding_api_key", ""))
        self.embedding_base_url.setText(config.get("embedding_base_url", ""))
        self.embedding_model.setText(config.get("embedding_model", ""))

        # Update account selector
        self.update_account_selector()

        # Select first account if available
        if self.accounts:
            self.current_account_index = 0
            self.account_selector.setCurrentIndex(0)
            self.select_account(0)
            self.delete_account_button.setEnabled(True)
        else:
            # If no accounts exist after trying to load, create a default one
            self.add_account()

        logger.info("Settings loaded from config.json.")

    def update_account_selector(self):
        """Update the account selector dropdown with current accounts."""
        self.account_selector.clear()
        for account in self.accounts:
            display_name = (
                f"{account['account_name']} ({account['imap_user']})"
                if account["imap_user"]
                else account["account_name"]
            )
            self.account_selector.addItem(display_name)

        # Enable/disable delete button based on number of accounts
        self.delete_account_button.setEnabled(len(self.accounts) > 1)

        logger.info(f"Account selector updated with {len(self.accounts)} accounts")

    def select_account(self, index):
        """Select an account from the dropdown and update UI with its settings."""
        if index < 0 or index >= len(self.accounts):
            return

        self.current_account_index = index
        account = self.accounts[index]

        # Update UI with selected account settings
        self.imap_host.setText(account.get("imap_host", ""))
        self.imap_user.setText(account.get("imap_user", ""))
        self.imap_pass.setText(account.get("imap_pass", ""))
        self.imap_port.setValue(account.get("imap_port", 993))
        self.use_ssl.setChecked(account.get("use_ssl", True))
        self.db_path.setText(account.get("db_path", ""))

        # Update current settings
        self.settings = account.copy()

        # If credentials are incomplete, skip any network connection and clear folders
        if not (
            account.get("imap_host")
            and account.get("imap_user")
            and account.get("imap_pass")
        ):
            self.folder_names = []
            self.populate_folders()
            logger.info(
                "Selected account is incomplete; skipping IMAP connection and folder retrieval."
            )
            return

        # Connect to server and populate folders
        mail = self.connect_imap(
            account.get("imap_host", ""),
            account.get("imap_user", ""),
            account.get("imap_pass", ""),
            account.get("imap_port", 993),
            account.get("use_ssl", True),
        )

        if mail:
            self.folder_names = get_folder_names(mail)
            mail.logout()
            self.populate_folders()

            # Check previously selected folders
            if "selected_folders" in account:
                for cb in self.folder_checkboxes:
                    cb.setChecked(cb.text() in account["selected_folders"])

        logger.info(f"Selected account: {account.get('account_name', 'unnamed')}")
        
        # Adjust minimum size in case the settings content has changed
        self.set_dpi_aware_minimum_size()

    def add_account(self):
        """Add a new account configuration."""
        new_account = {
            "account_name": f"New Account {len(self.accounts) + 1}",
            "imap_host": "",
            "imap_user": "",
            "imap_pass": "",
            "imap_port": 993,
            "use_ssl": True,
            "db_path": f"emails_{len(self.accounts) + 1}.db",
            "selected_folders": [],
        }
        self.accounts.append(new_account)

        # Update UI
        self.update_account_selector()
        self.current_account_index = len(self.accounts) - 1
        self.account_selector.setCurrentIndex(self.current_account_index)
        self.select_account(self.current_account_index)

        QMessageBox.information(
            self, "New Account", "Please configure the new account settings and save."
        )
        logger.info(f"New account added (total: {len(self.accounts)})")

    def delete_account(self):
        if self.current_account_index < 0 or len(self.accounts) <= 1:
            return

        account_name = self.accounts[self.current_account_index]["account_name"]
        reply = QMessageBox.question(
            self,
            "Delete Account",
            f"Are you sure you want to delete '{account_name}'?\nThis will not delete the database file.",
            QMessageBox.Yes | QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            db_path_to_delete = self.accounts[self.current_account_index]["db_path"]
            if db_path_to_delete in self.email_cache:
                del self.email_cache[db_path_to_delete]
                logger.info(f"Cache cleared for deleted account: {db_path_to_delete}")

            # Remove the account
            del self.accounts[self.current_account_index]

            # Update UI
            self.update_account_selector()
            if self.accounts:
                new_index = min(self.current_account_index, len(self.accounts) - 1)
                self.account_selector.setCurrentIndex(new_index)
            else:
                self.add_account()  # Add a new empty account if all were deleted

            self.save_settings()  # Update config file

            QMessageBox.information(
                self, "Account Deleted", f"Account '{account_name}' has been deleted."
            )

    def populate_folders(self):
        # Remove existing checkboxes
        for cb in self.folder_checkboxes:
            self.folder_layout.removeWidget(cb)
            cb.deleteLater()
        self.folder_checkboxes = []

        # Add new checkboxes
        for folder in self.folder_names:
            cb = QCheckBox(folder)
            # Check if it's Inbox or contains 'sent' in the name
            if folder.lower() == "inbox" or "sent" in folder.lower():
                cb.setChecked(True)
            self.folder_layout.addWidget(cb)
            self.folder_checkboxes.append(cb)
        
        # Adjust minimum size after folder list changes
        self.set_dpi_aware_minimum_size()

    def get_selected_folders(self):
        # Return the exact folder names as they appear in the folder list
        return [cb.text() for cb in self.folder_checkboxes if cb.isChecked()]

    def sync_emails(self):
        """Synchronize emails for all configured accounts"""
        if not self.accounts:
            QMessageBox.warning(
                self, "Error", "No accounts configured. Please add an account first."
            )
            return

        # Collect active accounts to sync
        accounts_to_sync = []
        for account in self.accounts:
            if account["imap_host"] and account["imap_user"] and account["imap_pass"]:
                selected_folders = account.get("selected_folders", [])
                if selected_folders:
                    accounts_to_sync.append(account)

        if not accounts_to_sync:
            QMessageBox.warning(
                self, "Error", "No accounts with selected folders to sync."
            )
            return

        self.query_progress.setVisible(True)
        self.query_progress.setValue(0)
        self.sync_button.setEnabled(False)
        self.deep_research_checkbox.setEnabled(False)
        self.background_button.setEnabled(False)

        # Get global AI settings from UI
        ai_settings = {
            "api_key": config.get("embedding_api_key"),
            "base_url": config.get("embedding_base_url"),
        }

        # Create a list to track all sync workers
        self.sync_workers = []
        self.sync_completed_count = 0
        self.sync_error_count = 0
        self.last_sync_error_message = ""
        self.total_sync_accounts = len(accounts_to_sync)
        self.set_sync_status("Syncing…")

        # Start a worker for each account
        for account in accounts_to_sync:
            # Use the parallel worker implementation
            worker = ParallelEmailSyncWorker(
                account, account["selected_folders"], ai_settings
            )
            worker.progress.connect(self.update_sync_progress)
            worker.finished.connect(self.account_sync_finished)
            worker.error.connect(self.sync_error)
            self.sync_workers.append(worker)
            worker.start()

    def update_sync_progress(self, value):
        # We'll use the average progress across all accounts
        if self.total_sync_accounts > 0:
            current_progress = sum(
                getattr(worker, "last_progress", 0) for worker in self.sync_workers
            )
            avg_progress = current_progress / self.total_sync_accounts
            self.query_progress.setValue(int(avg_progress))

    def account_sync_finished(self):
        self.sync_completed_count += 1
        self.emails_synced = True
        self._check_sync_completion()

    def sync_error(self, message):
        # A sync worker failed (e.g. no internet connection). Instead of a modal
        # pop-up per failing account, record it and surface it in the status line.
        self.sync_error_count += 1
        self.last_sync_error_message = message
        logger.warning(f"Email sync error: {message}")
        self._check_sync_completion()

    def set_sync_status(self, text):
        """Update the persistent sync status line at the bottom of the window."""
        if hasattr(self, "sync_status_label"):
            self.sync_status_label.setText(text)

    def _summarize_sync_error(self, message):
        """Turn a raw sync error into a short, user-friendly reason."""
        msg = (message or "").lower()
        connection_indicators = [
            "connect",
            "no connection",
            "timed out",
            "timeout",
            "network",
            "unreachable",
            "getaddrinfo",
            "name or service not known",
            "temporary failure in name resolution",
            "socket",
            "errno 11001",
            "errno 11002",
        ]
        if any(indicator in msg for indicator in connection_indicators):
            return "no connection"
        first_line = (message or "").strip().splitlines()
        short = first_line[0] if first_line else "unknown error"
        return short[:80]

    def _check_sync_completion(self):
        """Finalize the UI once every account has reported (success or failure)."""
        if (
            self.sync_completed_count + self.sync_error_count
        ) < self.total_sync_accounts:
            return

        # Re-enable controls now that all accounts have reported.
        self.sync_button.setEnabled(True)
        self.deep_research_checkbox.setEnabled(True)
        self.background_button.setEnabled(True)

        # Reload the cache from whatever was synced successfully (or previously
        # stored on disk if everything failed).
        self.populate_initial_cache()

        timestamp = time.strftime("%Y-%m-%d %H:%M")
        if self.sync_error_count == 0:
            self.query_progress.setValue(100)
            self.set_sync_status(f"Last sync: {timestamp}")
        else:
            self.query_progress.setValue(0)
            reason = self._summarize_sync_error(self.last_sync_error_message)
            if self.sync_completed_count > 0:
                self.set_sync_status(
                    f"Last sync: {timestamp} — {self.sync_completed_count}/"
                    f"{self.total_sync_accounts} ok, "
                    f"{self.sync_error_count} failed ({reason})"
                )
            else:
                self.set_sync_status(f"Last sync failed: {timestamp} — {reason}")

    def search_emails(self):
        query = self.query_input.text()
        if not query:
            QMessageBox.warning(self, "Error", "Please enter a query.")
            return

        if self.deep_research_checkbox.isChecked():
            self.deep_research(query)
            return

        # Get all valid accounts for searching across all of them
        valid_accounts = [
            account
            for account in self.accounts
            if account["imap_host"] and account["imap_user"] and account["db_path"]
        ]

        if not valid_accounts:
            QMessageBox.warning(self, "Error", "No valid accounts configured.")
            return

        self.sync_button.setEnabled(False)
        self.deep_research_checkbox.setEnabled(False)
        self.background_button.setEnabled(False)
        self.results_text_edit.clear()
        self.query_progress.setValue(0)
        self.query_progress.setVisible(True)

        # Set initial status message
        self.results_text_edit.setPlainText(
            f"Searching across {len(valid_accounts)} accounts for: {query}\n\nGenerating response..."
        )
        self.streaming_buffer = ""
        self.streaming_started = False  # Reset streaming flag

        # Get global AI settings from UI
        self.query_worker = QueryWorker(valid_accounts, query, self.email_cache)
        self.query_worker.finished.connect(self.display_response)
        self.query_worker.error.connect(self.query_error)
        self.query_worker.streaming_chunk.connect(self.display_streaming_chunk)
        self.query_worker.start()

    def deep_research(self, query: str):
        """
        Perform deep research on a query using a dedicated research agent.

        Args:
            query: The research query
        """
        if not query.strip():
            QMessageBox.warning(self, "Empty Query", "Please enter a research query.")
            return

        # Get all valid accounts
        valid_accounts = [
            account
            for account in self.accounts
            if account["imap_host"] and account["imap_user"] and account["db_path"]
        ]

        if not valid_accounts:
            QMessageBox.warning(self, "No Accounts", "No valid accounts configured.")
            return

        # Create a progress area in the response box
        self.results_text_edit.clear()
        self.results_text_edit.append("Research in Progress\n\n")
        self.results_text_edit.append(f"Analyzing your emails to answer: {query}\n")
        self.results_text_edit.append("Research Steps:\n")

        # Disable the search button and show a progress indicator
        self.sync_button.setEnabled(False)
        self.deep_research_checkbox.setEnabled(False)
        self.background_button.setEnabled(False)
        self.query_progress.setVisible(True)
        self.query_progress.setValue(0)

        # Create and start the research worker with all valid accounts
        self.research_worker = DeepResearchWorker(valid_accounts, query)
        self.research_worker.progress.connect(self.update_research_progress)
        self.research_worker.finished.connect(self.display_response)
        self.research_worker.error.connect(self.research_error)
        self.research_worker.start()

        # Log the start of research
        logger.info(
            f"Started deep research for query: {query} across {len(valid_accounts)} accounts"
        )

    def update_research_progress(self, message: str):
        """
        Update the progress display for the deep research process.

        Args:
            message: Progress message
        """
        # Add the progress message as a new line in the text output
        self.results_text_edit.append(f"• {message}")

        # Scroll to the bottom to show the latest progress
        self.results_text_edit.verticalScrollBar().setValue(
            self.results_text_edit.verticalScrollBar().maximum()
        )

        # Allow UI to update
        QApplication.processEvents()

        # Log progress message with sanitization to prevent Unicode encoding errors
        sanitized_message = sanitize_log_message(message)
        logger.info(f"Research progress: {sanitized_message}")

    def research_error(self, message: str):
        """Handle deep research errors."""
        sanitized_message = sanitize_log_message(message)
        logger.error(f"Research error: {sanitized_message}")
        QMessageBox.warning(
            self, "Research Error", f"Error during deep research: {message}"
        )
        self.sync_button.setEnabled(True)
        self.deep_research_checkbox.setEnabled(True)
        self.background_button.setEnabled(True)
        self.query_progress.setMaximum(100)
        self.query_progress.setValue(0)

    def query_error(self, message):
        QMessageBox.warning(
            self, "Query Error", f"Error during query processing: {message}"
        )
        self.sync_button.setEnabled(True)

    def display_streaming_chunk(self, chunk):
        """Handle streaming chunks from the query worker."""
        if not self.streaming_started:
            self.streaming_buffer = ""  # Clear buffer on first chunk
            self.streaming_started = True

        self.streaming_buffer += chunk
        self.display_formatted_content(self.streaming_buffer)

        # Scroll to the bottom to show the latest content
        self.results_text_edit.verticalScrollBar().setValue(
            self.results_text_edit.verticalScrollBar().maximum()
        )

    def display_response(self, response):
        """Display the final, complete response."""
        if response and isinstance(response, str) and len(response.strip()) > 0:
            self.display_formatted_content(response)

        # Handle completion tasks
        self.query_progress.setVisible(False)
        self.sync_button.setEnabled(True)
        self.deep_research_checkbox.setEnabled(True)
        self.background_button.setEnabled(True)

        # Scroll to the top of the response after completion
        self.results_text_edit.verticalScrollBar().setValue(0)

    def display_formatted_content(self, text: str):
        """Converts markdown text to HTML and displays it with proper styling."""
        html_response = markdown2.markdown(
            text, extras=["tables", "fenced-code-blocks", "strike"]
        )

        # Add CSS to ensure markdown styles are rendered correctly
        full_html = f"""
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
                color: #333333;
            }}
            h1, h2, h3, h4, h5, h6 {{
                font-weight: bold;
            }}
            h2 {{ font-size: 1.2em; margin-top: 1em; margin-bottom: 0.5em; }}
            h3 {{ font-size: 1.1em; margin-top: 1em; margin-bottom: 0.5em; }}
            strong, b {{
                font-weight: bold;
            }}
            em, i {{
                font-style: italic;
            }}
            ul, ol {{
                margin-left: 20px;
            }}
            li {{
                margin-bottom: 5px;
            }}
            code {{
                font-family: 'Courier New', Courier, monospace;
                background-color: #f0f0f0;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            pre {{
                display: block;
                font-family: 'Courier New', Courier, monospace;
                background-color: #f0f0f0;
                padding: 10px;
                border-radius: 3px;
                white-space: pre-wrap;
            }}
            hr {{
                border: none;
                border-top: 1px solid #ccc;
                margin: 1em 0;
            }}
        </style>
        {html_response}
        """
        self.results_text_edit.setHtml(full_html)

        # Re-enable buttons after search is completed
        self.sync_button.setEnabled(True)
        self.deep_research_checkbox.setEnabled(True)
        self.background_button.setEnabled(True)

    def save_settings_to_config(self):
        """Save all account settings to the config file."""
        save_config(config)

    def populate_initial_cache(self):
        """Load email data from all account databases into the cache."""
        logger.info("Populating email cache...")
        self.email_cache.clear()  # Start with a fresh cache
        for account in self.accounts:
            db_path = account.get("db_path")
            if db_path and os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path, check_same_thread=False)
                    c = conn.cursor()
                    c.execute(
                        "SELECT id, uid, folder, subject, from_addr, to_addr, date, body, attachment_names, has_attachments, embedding FROM emails"
                    )
                    emails_from_db = c.fetchall()
                    conn.close()

                    # Convert embeddings to numpy arrays and pre-calculate norms
                    email_entries = []
                    embedding_arrays = []
                    for row in emails_from_db:
                        if row[10]:
                            email_entries.append(row)
                            embedding_arrays.append(
                                np.frombuffer(row[10], dtype=np.float64)
                            )

                    if embedding_arrays:
                        email_embeddings_matrix = np.array(embedding_arrays)
                        email_norms = np.linalg.norm(email_embeddings_matrix, axis=1)
                        self.email_cache[db_path] = (
                            email_entries,
                            email_embeddings_matrix,
                            email_norms,
                        )
                        logger.info(
                            f"Cached {len(email_entries)} emails, embeddings, and norms for {account['account_name']}"
                        )
                    else:
                        self.email_cache[db_path] = ([], np.array([]), np.array([]))
                        logger.info(
                            f"No emails with embeddings to cache for {account['account_name']}"
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to populate cache for {account['account_name']}: {e}"
                    )
        logger.info("Email cache population complete.")

    def load_background_info(self):
        """Load background information from file if it exists."""
        try:
            if os.path.exists("background_info.txt"):
                with open("background_info.txt", "r", encoding="utf-8") as f:
                    self.background_info = f.read()
                logger.info("Background information loaded from background_info.txt")
            else:
                logger.info("No background_info.txt file found")
        except Exception as e:
            logger.error(f"Error loading background info: {e}")
            self.background_info = ""

    def generate_background_info(self):
        """Generate background information by analyzing emails across all accounts."""
        # Get all valid accounts
        valid_accounts = [
            account
            for account in self.accounts
            if account["imap_host"] and account["imap_user"] and account["db_path"]
        ]

        if not valid_accounts:
            QMessageBox.warning(
                self,
                "No Accounts",
                "No valid accounts configured for background analysis.",
            )
            return

        # Check if any account has emails
        has_emails = False
        for account in valid_accounts:
            if os.path.exists(account["db_path"]):
                try:
                    conn = sqlite3.connect(account["db_path"], check_same_thread=False)
                    c = conn.cursor()
                    c.execute("SELECT COUNT(*) FROM emails")
                    count = c.fetchone()[0]
                    conn.close()
                    if count > 0:
                        has_emails = True
                        break
                except:
                    continue

        if not has_emails:
            QMessageBox.warning(
                self,
                "No Emails",
                "No emails found in any account. Please sync emails first.",
            )
            return

        # Disable buttons and show progress
        self.sync_button.setEnabled(False)
        self.deep_research_checkbox.setEnabled(False)
        self.background_button.setEnabled(False)
        self.results_text_edit.clear()
        self.results_text_edit.append("Generating Background Profile\n")
        self.results_text_edit.append("This may take a few minutes...\n")

        # Start background analysis worker
        self.background_worker = BackgroundAnalysisWorker(valid_accounts)
        self.background_worker.progress.connect(self.update_background_progress)
        self.background_worker.finished.connect(self.background_analysis_finished)
        self.background_worker.error.connect(self.background_analysis_error)
        self.background_worker.start()

        logger.info(f"Started background analysis for {len(valid_accounts)} accounts")

    def update_background_progress(self, message: str):
        """Update the progress display for background analysis."""
        self.results_text_edit.append(f"• {message}")
        self.results_text_edit.verticalScrollBar().setValue(
            self.results_text_edit.verticalScrollBar().maximum()
        )
        QApplication.processEvents()

    def background_analysis_finished(self, background_info: str):
        """Handle completion of background analysis."""
        self.background_info = background_info
        self.results_text_edit.clear()
        self.results_text_edit.setPlainText(background_info)

        # Re-enable buttons
        self.sync_button.setEnabled(True)
        self.deep_research_checkbox.setEnabled(True)
        self.background_button.setEnabled(True)

        QMessageBox.information(
            self,
            "Background Analysis Complete",
            "Background profile has been generated and saved to background_info.txt",
        )

        logger.info("Background analysis completed successfully")

    def background_analysis_error(self, message: str):
        """Handle background analysis errors."""
        logger.error(f"Background analysis error: {message}")
        QMessageBox.warning(
            self,
            "Background Analysis Error",
            f"Error during background analysis: {message}",
        )

        # Re-enable buttons
        self.sync_button.setEnabled(True)
        self.deep_research_checkbox.setEnabled(True)
        self.background_button.setEnabled(True)

    def init_db(self, db_path):
        conn = sqlite3.connect(db_path, check_same_thread=False)
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS emails (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        uid TEXT,
                        folder TEXT,
                        subject TEXT,
                        from_addr TEXT,
                        to_addr TEXT,
                        date TEXT,
                        body TEXT,
                        attachment_names TEXT,
                        has_attachments INTEGER DEFAULT 0,
                        embedding BLOB,
                        UNIQUE(uid, folder)
                    )"""
        )
        conn.commit()
        logger.info("Database initialized for main thread.")
        return conn


def main():
    try:
        # Enable DPI scaling support for Windows
        if hasattr(Qt, "AA_EnableHighDpiScaling"):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, "AA_UseHighDpiPixmaps"):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        app = QApplication(sys.argv)

        # Set application properties for better Windows integration
        app.setApplicationName("NeuralMail")
        app.setApplicationDisplayName("NeuralMail")
        app.setApplicationVersion("1.1")
        app.setOrganizationName("NeuralMail")
        app.setOrganizationDomain("neuralmail.app")

        # Set the application icon at the QApplication level
        if os.path.exists("icon.ico"):
            app_icon = QIcon("icon.ico")
            app.setWindowIcon(app_icon)

        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        # Show error in GUI if possible
        try:
            QMessageBox.critical(
                None,
                "Fatal Error",
                f"A fatal error occurred:\n{str(e)}\n\nCheck neuralmail.log for details.",
            )
        except:
            print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
