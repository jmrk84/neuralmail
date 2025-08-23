import logging
import sqlite3
import imaplib
from email.header import decode_header
import numpy as np
import openai
import tiktoken
import time
from .config import config

logger = logging.getLogger(__name__)


def decode_email_header(header):
    """Safely decodes an email header, which can be a string or Header object."""
    if not header:
        return ""

    try:
        decoded_parts = decode_header(header)
        header_text = []
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                try:
                    header_text.append(
                        part.decode(encoding if encoding else "utf-8", errors="replace")
                    )
                except (UnicodeDecodeError, LookupError):
                    header_text.append(part.decode("latin1", errors="replace"))
            else:
                header_text.append(str(part))
        return "".join(header_text)
    except Exception as e:
        logger.error(f"Failed to decode header: {e}")
        return str(header)


def connect_imap(imap_host, imap_user, imap_pass, imap_port=993, use_ssl=True):
    """Connect to the IMAP server."""
    try:
        if use_ssl:
            mail = imaplib.IMAP4_SSL(imap_host, imap_port)
        else:
            mail = imaplib.IMAP4(imap_host, imap_port)
        mail.login(imap_user, imap_pass)
        logger.info(f"Successfully connected to IMAP server {imap_host}.")
        return mail
    except imaplib.IMAP4.error as e:
        logger.error(f"IMAP login failed for {imap_user}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error connecting to IMAP server {imap_host}: {e}")
        return None


def init_db(db_path):
    """Initialize the SQLite database."""
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
    logger.info(f"Database initialized: {db_path}")
    return conn


def get_embedding(
    text: str, client: openai.OpenAI, model: str = None, max_tokens: int = None
) -> np.ndarray:
    """Get embedding for text using OpenAI API with retry logic."""
    model = model or config.get("embedding_model")
    max_tokens = max_tokens or config.get("max_tokens")

    try:
        text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
        text = text.replace("\\n", " ")

        # Handle tokenization for different model providers
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for non-OpenAI models (like Google Gemini)
            # Use cl100k_base encoding as a reasonable default
            encoding = tiktoken.get_encoding("cl100k_base")
            logger.warning(
                f"Embedding model '{model}' not found in tiktoken, using cl100k_base encoding as fallback"
            )

        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = encoding.decode(tokens)

        max_retries = 5
        for i in range(max_retries):
            try:
                response = client.embeddings.create(input=[text], model=model)
                return np.array(response.data[0].embedding)
            except (openai.APIConnectionError, openai.RateLimitError) as e:
                logger.warning(
                    f"Embedding API error ({type(e).__name__}), retry {i+1}/{max_retries}..."
                )
                if i < max_retries - 1:
                    time.sleep(2**i)  # Exponential backoff
                else:
                    raise
            except (openai.InvalidRequestError, openai.APIError) as e:
                logger.error(f"Fatal embedding API error: {e}")
                raise
        raise Exception("Failed to get embedding after multiple retries.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_embedding: {e}")
        raise


def openai_api_client(api_key, base_url=None):
    """Set up the OpenAI client using the API key and optional base URL from settings."""
    return openai.OpenAI(
        api_key=api_key, base_url=base_url if base_url else "https://api.openai.com/v1"
    )


def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)
