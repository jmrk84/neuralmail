"""
NeuralMail - AI-powered email search and analysis tool.

A battle-tested email archeology application that allows you to search and compile 
reports from multiple accounts with advanced AI capabilities.

Features:
- Multiple IMAP Account Support
- Smart Search using natural language
- Attachment Processing (PDF, DOCX)
- Deep Research Mode with agent-based analysis
- User profile creation for improved retrieval
- Parallel processing for efficiency
- Modern Qt-based UI
- Local storage for privacy

License: MIT
"""

__version__ = "1.0.0"
__author__ = "NeuralMail Team"
__email__ = "contact@neuralmail.app"

# Import main classes for easier access
from .main import MainWindow

__all__ = ["MainWindow", "__version__"]
