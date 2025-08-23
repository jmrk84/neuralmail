import logging
from PyQt5.QtCore import pyqtSignal, QThread
from .email_processor import EmailProcessor
from .utils import openai_api_client, connect_imap

logger = logging.getLogger(__name__)


class ParallelEmailSyncWorker(QThread):
    """
    A QThread worker that uses the parallel email processor to sync emails.
    This maintains the same interface as the original EmailSyncWorker class,
    but uses multiple threads internally for faster processing.
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, settings, folders, ai_settings=None):
        super().__init__()
        self.settings = settings
        self.folders = folders
        self.ai_settings = ai_settings
        self.last_progress = 0
        self.processor = None

    def run(self):
        try:
            if not self.ai_settings or not self.ai_settings.get("api_key"):
                self.error.emit("AI settings with API key are required for email sync.")
                return

            # Create OpenAI client from global settings
            client = openai_api_client(
                self.ai_settings["api_key"], self.ai_settings.get("base_url") or None
            )

            # Replace any "Sent" folder with "Sent Items" if "Sent" doesn't exist
            corrected_folders = []
            mail = None
            try:
                mail = connect_imap(
                    self.settings["imap_host"],
                    self.settings["imap_user"],
                    self.settings["imap_pass"],
                    self.settings["imap_port"],
                    self.settings["use_ssl"],
                )
                if mail:
                    # Get available folders
                    result, folders = mail.list()
                    available_folder_names = []
                    if result == "OK":
                        for folder in folders:
                            folder_info = folder.decode()
                            parts = folder_info.split(' "/" ')
                            if len(parts) == 2:
                                folder_name = parts[1].strip('"')
                                available_folder_names.append(folder_name)

                    # Check if "Sent" folder exists
                    for folder in self.folders:
                        if folder == "Sent" and "Sent" not in available_folder_names:
                            # Look for alternatives
                            for available in available_folder_names:
                                if "sent" in available.lower():
                                    corrected_folders.append(available)
                                    break
                            else:
                                corrected_folders.append(
                                    folder
                                )  # Keep original if no alternative found
                        else:
                            corrected_folders.append(folder)

                    mail.logout()
                else:
                    corrected_folders = (
                        self.folders
                    )  # Use original folders if connection fails
            except Exception as e:
                logger.error(f"Error checking folders: {e}")
                corrected_folders = self.folders
            finally:
                if mail:
                    try:
                        mail.logout()
                    except:
                        pass

            # Create and start the email processor with corrected folders
            self.processor = EmailProcessor(
                account=self.settings,
                folders=corrected_folders,
                progress_callback=self.update_progress,
            )

            # Start the processor and get the number of emails to sync
            email_count = self.processor.start(client)

            if email_count > 0:
                # Wait for completion
                success = self.processor.wait_completion()
                if success:
                    logger.info(
                        f'Email synchronization completed successfully for account {self.settings["account_name"]}.'
                    )
                else:
                    logger.warning(
                        f'Email synchronization completed with errors for account {self.settings["account_name"]}.'
                    )
            else:
                logger.info(
                    f'No emails to sync for account {self.settings["account_name"]}.'
                )

            self.finished.emit()

        except Exception as e:
            logger.error(
                f'Error during parallel email sync for account {self.settings["account_name"]}: {e}'
            )
            self.error.emit(str(e))

    def update_progress(self, value):
        """Callback function to update progress."""
        self.last_progress = value
        self.progress.emit(value)
