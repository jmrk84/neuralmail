import logging
import sqlite3
import threading
import queue
import time
import email
from email.header import decode_header
import numpy as np
import html2text
from concurrent.futures import ThreadPoolExecutor

from .utils import decode_email_header, connect_imap, init_db, get_embedding

logger = logging.getLogger(__name__)

# Constants
MAX_WORKERS = 10  # Maximum number of concurrent worker threads


class EmailProcessor:
    """
    A class to handle parallel processing of emails while ensuring safe database operations.
    Uses a producer-consumer pattern with a thread pool for fetching and processing emails,
    and a single database writer thread to avoid SQLite concurrency issues.
    """

    def __init__(
        self, account, folders, progress_callback=None, max_workers=MAX_WORKERS
    ):
        """
        Initialize the email processor.

        Args:
            account: Dictionary containing account settings
            folders: List of folders to sync
            progress_callback: Function to call with progress updates
            max_workers: Maximum number of concurrent worker threads
        """
        self.account = account
        self.folders = folders
        self.progress_callback = progress_callback
        self.max_workers = max_workers

        # Queues for communication between threads
        self.email_queue = queue.Queue(maxsize=100)  # Queue for processed emails
        self.progress_queue = queue.Queue()  # Queue for progress updates
        self.error_queue = queue.Queue()  # Queue for errors

        # Tracking variables
        self.total_emails_to_process = 0
        self.processed_count = 0
        self.emails_to_sync = []
        self.is_cancelled = False
        self.is_running = False

        # Threading objects
        self.db_thread = None
        self.progress_thread = None
        self.worker_pool = None
        self.lock = threading.Lock()

    def connect_imap(
        self, imap_host, imap_user, imap_pass, imap_port=993, use_ssl=True
    ):
        """Connect to the IMAP server."""
        return connect_imap(imap_host, imap_user, imap_pass, imap_port, use_ssl)

    def init_db(self, db_path):
        """Initialize the SQLite database."""
        return init_db(db_path)

    def get_embedding(self, text, client):
        """Get an OpenAI embedding for the text."""
        return get_embedding(text, client)

    def collect_emails_to_sync(self):
        """Collect list of emails that need to be synced."""
        mail = self.connect_imap(
            self.account["imap_host"],
            self.account["imap_user"],
            self.account["imap_pass"],
            self.account["imap_port"],
            self.account["use_ssl"],
        )

        if not mail:
            raise Exception(
                f"Failed to connect to IMAP server for account {self.account['account_name']}"
            )

        # Connect to database to check which emails we already have
        conn = self.init_db(self.account["db_path"])
        c = conn.cursor()

        self.emails_to_sync = []

        # For each folder, find emails that need syncing
        for folder in self.folders:
            try:
                # Select the folder - ensure proper quoting for folder names with spaces
                folder_name = f'"{folder}"'  # Always quote the folder name
                result, data = mail.select(folder_name, readonly=True)
                if result != "OK":
                    logger.error(f"Failed to select folder {folder_name}: {data}")
                    continue

                # Search for all emails
                result, data = mail.uid("search", None, "ALL")
                if result != "OK" or not data or not data[0]:
                    logger.error(
                        f'Failed to search emails in folder "{folder}": {data}'
                    )
                    continue

                uid_list = data[0].split() if data[0] else []

                # Check which emails are not yet in the database
                for uid in uid_list:
                    try:
                        uid_str = uid.decode("utf-8")
                        c.execute(
                            "SELECT 1 FROM emails WHERE uid=? AND folder=?",
                            (uid_str, folder),
                        )
                        if not c.fetchone():
                            self.emails_to_sync.append((uid_str, folder))
                    except Exception as e:
                        logger.error(f"Error checking UID {uid} in database: {e}")
            except Exception as e:
                logger.error(f'Error processing folder "{folder}": {e}')

        mail.logout()
        conn.close()

        logger.info(
            f'Found {len(self.emails_to_sync)} emails to sync for account {self.account["account_name"]}'
        )
        self.total_emails_to_process = len(self.emails_to_sync)
        return self.total_emails_to_process

    def process_email(self, uid, folder, client):
        """
        Process a single email: fetch from server, extract data, create embedding.
        Returns a tuple of (successful, email_data) where email_data contains all fields
        needed for database insertion.
        """
        mail = None
        try:
            # Connect to IMAP server
            mail = self.connect_imap(
                self.account["imap_host"],
                self.account["imap_user"],
                self.account["imap_pass"],
                self.account["imap_port"],
                self.account["use_ssl"],
            )

            if not mail:
                return (
                    False,
                    f"Failed to connect to IMAP server for UID {uid} in folder {folder}",
                )

            # Select the folder - ensure proper quoting for folder names with spaces
            folder_name = f'"{folder}"'  # Always quote the folder name
            result, data = mail.select(folder_name, readonly=True)
            if result != "OK":
                return False, f"Failed to select folder {folder_name}: {data}"

            # Fetch the email
            result, msg_data = mail.uid("fetch", uid, "(RFC822)")
            if result != "OK" or not msg_data or not msg_data[0]:
                return (
                    False,
                    f'Failed to fetch email UID {uid} in folder "{folder}": {msg_data}',
                )

            if not isinstance(msg_data[0][1], bytes):
                return (
                    False,
                    f'Invalid message data type for UID {uid} in folder "{folder}"',
                )

            # Parse the email
            email_message = email.message_from_bytes(msg_data[0][1])

            # Decode headers safely
            subject = decode_email_header(email_message.get("Subject", ""))
            from_addr = decode_email_header(email_message.get("From", ""))
            to_addr = decode_email_header(email_message.get("To", ""))
            date = email_message.get("Date", "")

            # Extract body and attachments
            body = ""
            attachment_names = []
            has_attachments = 0

            try:
                if email_message.is_multipart():
                    for part in email_message.walk():
                        if part.get_content_maintype() == "multipart":
                            continue
                        elif part.get_content_maintype() == "text":
                            try:
                                charset = part.get_content_charset()
                                payload = part.get_payload(decode=True)
                                if payload:
                                    if part.get_content_type() == "text/html":
                                        html_content = payload.decode(
                                            charset if charset else "utf-8",
                                            errors="ignore",
                                        )
                                        text_maker = html2text.HTML2Text()
                                        text_maker.ignore_links = True
                                        body += text_maker.handle(html_content)
                                    else:
                                        body += payload.decode(
                                            charset if charset else "utf-8",
                                            errors="ignore",
                                        )
                            except Exception as e:
                                logger.error(f"Error decoding part for UID {uid}: {e}")
                        elif "attachment" in str(part.get("Content-Disposition", "")):
                            filename = part.get_filename()
                            if filename:
                                attachment_names.append(filename)
                                has_attachments = 1
                else:
                    try:
                        payload = email_message.get_payload(decode=True)
                        if payload:
                            charset = email_message.get_content_charset()
                            if email_message.get_content_type() == "text/html":
                                html_content = payload.decode(
                                    charset if charset else "utf-8", errors="ignore"
                                )
                                text_maker = html2text.HTML2Text()
                                text_maker.ignore_links = True
                                body += text_maker.handle(html_content)
                            else:
                                body += payload.decode(
                                    charset if charset else "utf-8", errors="ignore"
                                )
                    except Exception as e:
                        logger.error(
                            f"Error decoding non-multipart message for UID {uid}: {e}"
                        )
            except Exception as e:
                logger.error(f"Error processing message structure for UID {uid}: {e}")
                return False, f"Error processing message structure: {e}"

            # Clean up body text
            try:
                lines = body.split("\n")
                cleaned_lines = [
                    line for line in lines if not line.lstrip().startswith(">")
                ]
                body = "\n".join(cleaned_lines)
                body = "\n".join(line for line in body.split("\n") if line.strip())
                body = body.rstrip("\n")

                # Clean surrogate pairs and other problematic characters
                body = body.encode("utf-8", "ignore").decode("utf-8")
            except Exception as e:
                logger.error(f"Error cleaning body text for UID {uid}: {e}")
                return False, f"Error cleaning body text: {e}"

            # Create embedding
            try:
                text_for_embedding = f"""
From: {from_addr}
To: {to_addr}
Date: {date}
Subject: {subject}

{body}
"""
                # Ensure the text is clean for embedding
                text_for_embedding = text_for_embedding.encode(
                    "utf-8", "ignore"
                ).decode("utf-8")
                embedding = self.get_embedding(text_for_embedding, client)
            except Exception as e:
                logger.error(f"Failed to generate embedding for UID {uid}: {e}")
                return False, f"Failed to generate embedding: {e}"

            # Prepare data for database insertion
            email_data = (
                uid,
                folder,
                subject,
                from_addr,
                to_addr,
                date,
                body,
                ",".join(attachment_names),
                has_attachments,
                embedding.tobytes(),
            )

            return True, email_data

        except Exception as e:
            return False, f"Error processing email UID {uid} in folder {folder}: {e}"
        finally:
            if mail:
                try:
                    mail.logout()
                except:
                    pass

    def worker_function(self, uid_folder_pair, client):
        """Worker function for the thread pool."""
        if self.is_cancelled:
            return

        uid, folder = uid_folder_pair
        success, result = self.process_email(uid, folder, client)

        if success:
            # Put the processed email in the queue for database insertion
            self.email_queue.put((uid, folder, result))
        else:
            # Log the error
            logger.error(result)
            self.error_queue.put(result)

        # Update progress
        with self.lock:
            self.processed_count += 1
            progress = int((self.processed_count / self.total_emails_to_process) * 100)
            self.progress_queue.put(progress)

    def db_writer_thread(self):
        """Thread function for writing to the database."""
        conn = self.init_db(self.account["db_path"])
        c = conn.cursor()

        while self.is_running:
            try:
                # Get the next processed email with a timeout
                try:
                    uid, folder, email_data = self.email_queue.get(timeout=1)
                except queue.Empty:
                    # If the queue is empty and all emails have been processed, exit
                    if self.processed_count >= self.total_emails_to_process:
                        break
                    continue

                try:
                    # Insert into database
                    c.execute(
                        """INSERT INTO emails 
                                (uid, folder, subject, from_addr, to_addr, date, body, 
                                attachment_names, has_attachments, embedding)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        email_data,
                    )
                    conn.commit()
                    logger.info(
                        f'Successfully synced email UID {uid} in folder "{folder}".'
                    )
                except Exception as e:
                    logger.error(f"Failed to insert email UID {uid} into database: {e}")

                # Mark task as done
                self.email_queue.task_done()

            except Exception as e:
                logger.error(f"Error in database writer thread: {e}")

        # Close database connection
        conn.close()
        logger.info(
            f"Database writer thread finished for account {self.account['account_name']}"
        )

    def progress_reporter_thread(self):
        """Thread function for reporting progress."""
        last_progress = 0

        while self.is_running:
            try:
                # Get the next progress update with a timeout
                try:
                    progress = self.progress_queue.get(timeout=1)
                except queue.Empty:
                    # If we've processed all emails, exit
                    if self.processed_count >= self.total_emails_to_process:
                        break
                    continue

                # Only report if progress has changed significantly (every 5%)
                if progress >= last_progress + 5 or progress >= 100:
                    last_progress = progress
                    if self.progress_callback:
                        self.progress_callback(progress)
                    logger.info(
                        f"Progress: {progress}% ({self.processed_count}/{self.total_emails_to_process})"
                    )

                # Mark task as done
                self.progress_queue.task_done()

            except Exception as e:
                logger.error(f"Error in progress reporter thread: {e}")

    def start(self, client):
        """
        Start the email processing operation with a thread pool.

        Args:
            client: The OpenAI client to use for embeddings

        Returns:
            int: The number of emails to be synced
        """
        if self.is_running:
            logger.warning("Email processor is already running")
            return 0

        # Collect emails to sync
        email_count = self.collect_emails_to_sync()
        if email_count == 0:
            logger.info(f"No emails to sync for account {self.account['account_name']}")
            return 0

        # Reset state
        self.processed_count = 0
        self.is_cancelled = False
        self.is_running = True

        # Start the database writer thread
        self.db_thread = threading.Thread(target=self.db_writer_thread)
        self.db_thread.daemon = True
        self.db_thread.start()

        # Start the progress reporter thread
        self.progress_thread = threading.Thread(target=self.progress_reporter_thread)
        self.progress_thread.daemon = True
        self.progress_thread.start()

        # Create and start the worker pool
        self.worker_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        # Submit tasks to the worker pool
        for uid_folder_pair in self.emails_to_sync:
            self.worker_pool.submit(self.worker_function, uid_folder_pair, client)

        logger.info(
            f"Started processing {email_count} emails with {self.max_workers} workers"
        )
        return email_count

    def wait_completion(self):
        """Wait for all tasks to complete."""
        if not self.is_running:
            return True

        # Shutdown the worker pool (wait for all tasks to complete)
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)

        # Wait for all emails to be processed
        self.email_queue.join()

        # Wait for the threads to finish
        self.is_running = False
        if self.db_thread:
            self.db_thread.join()
        if self.progress_thread:
            self.progress_thread.join()

        # Report completion
        logger.info(
            f"Completed processing {self.processed_count}/{self.total_emails_to_process} emails"
        )

        # Check if there were any errors
        errors = []
        while not self.error_queue.empty():
            errors.append(self.error_queue.get())

        return len(errors) == 0

    def cancel(self):
        """Cancel the email processing operation."""
        self.is_cancelled = True
        self.is_running = False

        # Shutdown the worker pool
        if self.worker_pool:
            self.worker_pool.shutdown(wait=False)

        logger.info("Email processing cancelled")
