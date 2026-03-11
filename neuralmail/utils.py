import logging
import sqlite3
import imaplib
from email.header import decode_header
import numpy as np
import openai
import tiktoken
import time
import requests
import json
from .config import config

logger = logging.getLogger(__name__)

# Initialize tokenizer for input truncation
try:
    encoding = tiktoken.encoding_for_model("gpt-4")
except KeyError:
    encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    return len(encoding.encode(text))

def truncate_messages_for_context(messages, max_tokens_for_response: int = 4000):
    """
    Truncate messages to fit within the user's context limit minus response tokens.
    
    Args:
        messages: List of message dictionaries
        max_tokens_for_response: Tokens to reserve for the response
        
    Returns:
        Truncated messages that fit within the limit
    """
    # Get user's max context setting
    user_max_context = config.get("llm_max_context", 256000)
    
    # Calculate available tokens for input (subtract response tokens)
    available_tokens = user_max_context - max_tokens_for_response
    
    # Count current tokens
    current_tokens = sum(count_tokens(msg.get("content", "")) for msg in messages)
    
    if current_tokens <= available_tokens:
        return messages  # No truncation needed
    
    logger.warning(f"Input tokens ({current_tokens}) exceed limit ({available_tokens}), truncating...")
    
    # Truncate from the end of the longest message (usually user content)
    truncated_messages = []
    remaining_tokens = available_tokens
    
    for msg in messages:
        content = msg.get("content", "")
        content_tokens = count_tokens(content)
        
        if content_tokens <= remaining_tokens:
            # Message fits entirely
            truncated_messages.append(msg)
            remaining_tokens -= content_tokens
        else:
            # Truncate this message
            if remaining_tokens > 100:  # Keep at least some content
                # Calculate how many characters to keep (rough approximation)
                chars_per_token = len(content) / content_tokens if content_tokens > 0 else 4
                max_chars = int(remaining_tokens * chars_per_token * 0.9)  # 90% safety margin
                
                truncated_content = content[:max_chars] + "... [truncated due to token limit]"
                truncated_msg = msg.copy()
                truncated_msg["content"] = truncated_content
                truncated_messages.append(truncated_msg)
                
            remaining_tokens = 0  # Used up all tokens
            break
    
    final_tokens = sum(count_tokens(msg.get("content", "")) for msg in truncated_messages)
    logger.info(f"Truncated input from {current_tokens} to {final_tokens} tokens")
    
    return truncated_messages


def prepare_openrouter_request_params(base_url: str, model: str, base_params: dict) -> dict:
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


class OpenRouterClient:
    """
    Wrapper for OpenAI client that handles OpenRouter-specific parameters like 'provider'.
    """
    def __init__(self, api_key, base_url=None):
        self.base_url = base_url or "https://api.openai.com/v1"
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key, base_url=self.base_url)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the underlying OpenAI client."""
        return getattr(self.client, name)
    
    @property 
    def chat(self):
        """Return a chat wrapper that handles provider parameters."""
        return ChatWrapper(self.client.chat, self.base_url)


class ChatWrapper:
    """Wrapper for chat completions that handles OpenRouter provider parameters."""
    
    def __init__(self, chat_client, base_url):
        self.chat_client = chat_client
        self.base_url = base_url
    
    def __getattr__(self, name):
        """Delegate all other attributes to the underlying chat client."""
        return getattr(self.chat_client, name)
    
    @property
    def completions(self):
        """Return completions wrapper."""
        return CompletionsWrapper(self.chat_client.completions, self.base_url)


class CompletionsWrapper:
    """Wrapper for completions that handles OpenRouter provider parameters."""
    
    def __init__(self, completions_client, base_url):
        self.completions_client = completions_client
        self.base_url = base_url
    
    def __getattr__(self, name):
        """Delegate all other attributes to the underlying completions client."""
        return getattr__(self.completions_client, name)
    
    def create(self, **kwargs):
        """
        Create completion with OpenRouter provider support.
        If provider parameter is present and we're using OpenRouter, handle it specially.
        """
        # Check if we're using OpenRouter and have provider parameter
        if (self.base_url and "openrouter.ai" in self.base_url and 
            "provider" in kwargs):
            
            # Extract provider parameter before creating request
            provider = kwargs.pop("provider")
            is_streaming = kwargs.get("stream", False)
            
            # Get the API key from the client
            try:
                # Try different ways to access the API key
                if hasattr(self.completions_client, '_client'):
                    api_key = self.completions_client._client.api_key
                elif hasattr(self.completions_client, 'api_key'):
                    api_key = self.completions_client.api_key
                else:
                    # Fallback - traverse the client hierarchy
                    client = self.completions_client
                    while hasattr(client, '_client') and not hasattr(client, 'api_key'):
                        client = client._client
                    api_key = client.api_key
            except (AttributeError, KeyError):
                raise openai.APIError("Could not access API key from OpenAI client")
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare request body
            body = kwargs.copy()
            body["provider"] = provider
            
            if is_streaming:
                # Handle streaming response
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    stream=True,
                    timeout=60
                )
                
                if response.status_code != 200:
                    raise openai.APIError(f"OpenRouter API error: {response.status_code} - {response.text}")
                
                return self._handle_streaming_response(response)
            else:
                # Handle non-streaming response
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=60
                )
                
                if response.status_code != 200:
                    raise openai.APIError(f"OpenRouter API error: {response.status_code} - {response.text}")
                
                return self._create_response_object(response.json())
        else:
            # Use normal OpenAI client for non-OpenRouter requests or requests without provider
            return self.completions_client.create(**kwargs)
    
    def _create_response_object(self, response_data):
        """Create a mock OpenAI response object from OpenRouter response data."""
        from types import SimpleNamespace
        
        # Create usage object
        usage_data = response_data.get("usage", {})
        usage = SimpleNamespace(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        # Create choice objects
        choices = []
        for choice_data in response_data.get("choices", []):
            message = SimpleNamespace(
                content=choice_data.get("message", {}).get("content", ""),
                role="assistant"
            )
            choice = SimpleNamespace(
                message=message,
                finish_reason=choice_data.get("finish_reason"),
                index=choice_data.get("index", 0)
            )
            choices.append(choice)
        
        # Create response object
        return SimpleNamespace(
            choices=choices,
            usage=usage,
            id=response_data.get("id"),
            model=response_data.get("model"),
            object="chat.completion"
        )
    
    def _handle_streaming_response(self, response):
        """Handle streaming response from OpenRouter."""
        from types import SimpleNamespace
        
        def stream_generator():
            usage_info = None
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                        
                        if line.strip() == '[DONE]':
                            break
                        
                        try:
                            chunk_data = json.loads(line)
                            
                            # Handle usage info in streaming
                            if 'usage' in chunk_data and chunk_data['usage']:
                                usage_info = chunk_data['usage']
                            
                            # Create chunk object
                            choices = []
                            for choice_data in chunk_data.get("choices", []):
                                delta_data = choice_data.get("delta", {})
                                delta = SimpleNamespace(
                                    content=delta_data.get("content"),
                                    role=delta_data.get("role")
                                )
                                choice = SimpleNamespace(
                                    delta=delta,
                                    finish_reason=choice_data.get("finish_reason"),
                                    index=choice_data.get("index", 0)
                                )
                                choices.append(choice)
                            
                            chunk = SimpleNamespace(
                                choices=choices,
                                id=chunk_data.get("id"),
                                model=chunk_data.get("model"),
                                object="chat.completion.chunk"
                            )
                            
                            # Add usage info if present
                            if usage_info:
                                chunk.usage = SimpleNamespace(
                                    prompt_tokens=usage_info.get("prompt_tokens", 0),
                                    completion_tokens=usage_info.get("completion_tokens", 0),
                                    total_tokens=usage_info.get("total_tokens", 0)
                                )
                            
                            yield chunk
                            
                        except json.JSONDecodeError:
                            continue
        
        return stream_generator()


def openai_api_client(api_key, base_url=None):
    """Set up the OpenAI/OpenRouter client using the API key and optional base URL from settings."""
    return OpenRouterClient(api_key, base_url)


def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)
