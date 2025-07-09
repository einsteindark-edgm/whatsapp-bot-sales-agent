"""
CLI Spinner Implementation.

This module provides a threaded spinner for visual feedback during
async operations in the CLI interface.
"""

import sys
import threading
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class SpinnerStyle(Enum):
    """Available spinner styles."""

    DOTS = "dots"
    BARS = "bars"
    ARROWS = "arrows"
    CLOCK = "clock"
    MOON = "moon"
    BOUNCE = "bounce"


@dataclass
class SpinnerConfig:
    """Configuration for spinner behavior."""

    style: SpinnerStyle = SpinnerStyle.DOTS
    delay: float = 0.1
    message: str = "Processing..."
    success_message: str = "Done!"
    error_message: str = "Error!"
    color_enabled: bool = True

    # Animation sequences
    sequences = {
        SpinnerStyle.DOTS: ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        SpinnerStyle.BARS: ["|", "/", "-", "\\"],
        SpinnerStyle.ARROWS: ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
        SpinnerStyle.CLOCK: [
            "🕐",
            "🕑",
            "🕒",
            "🕓",
            "🕔",
            "🕕",
            "🕖",
            "🕗",
            "🕘",
            "🕙",
            "🕚",
            "🕛",
        ],
        SpinnerStyle.MOON: ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"],
        SpinnerStyle.BOUNCE: ["⠁", "⠂", "⠄", "⠂"],
    }


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class SpinnerThread(threading.Thread):
    """
    Threaded spinner for non-blocking UI feedback.

    This class provides a spinner that runs in a separate thread,
    allowing async operations to proceed without blocking the UI.
    """

    def __init__(self, config: Optional[SpinnerConfig] = None):
        """
        Initialize the spinner thread.

        Args:
            config: Optional spinner configuration
        """
        super().__init__(daemon=True)
        self.config = config or SpinnerConfig()
        self._stop_event = threading.Event()
        self._success = False
        self._error = False
        self._error_message = ""
        self._current_message = self.config.message

        # Get animation sequence
        self.sequence = self.config.sequences.get(
            self.config.style, self.config.sequences[SpinnerStyle.DOTS]
        )

        # Check if colors are supported
        self._colors_supported = (
            self.config.color_enabled and hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        )

    def _colorize(self, text: str, color: str) -> str:
        """
        Colorize text if colors are supported.

        Args:
            text: Text to colorize
            color: ANSI color code

        Returns:
            Colorized text or original text if colors not supported
        """
        if self._colors_supported:
            return f"{color}{text}{Colors.RESET}"
        return text

    def _write_line(self, text: str):
        """
        Write a line to stdout with proper formatting.

        Args:
            text: Text to write
        """
        # Clear the line
        sys.stdout.write("\r\033[K")
        sys.stdout.write(text)
        sys.stdout.flush()

    def run(self):
        """Run the spinner animation."""
        index = 0

        while not self._stop_event.is_set():
            # Get current spinner character
            spinner_char = self.sequence[index % len(self.sequence)]

            # Format the spinner line
            if self._colors_supported:
                spinner_display = self._colorize(spinner_char, Colors.CYAN)
                message_display = self._colorize(self._current_message, Colors.WHITE)
            else:
                spinner_display = spinner_char
                message_display = self._current_message

            # Write spinner line
            line = f"{spinner_display} {message_display}"
            self._write_line(line)

            # Wait for next frame
            if self._stop_event.wait(self.config.delay):
                break

            index += 1

        # Show final message
        self._show_final_message()

    def _show_final_message(self):
        """Show the final message based on completion status."""
        if self._success:
            if self._colors_supported:
                icon = self._colorize("✓", Colors.GREEN)
                message = self._colorize(self.config.success_message, Colors.GREEN)
            else:
                icon = "✓"
                message = self.config.success_message

            final_line = f"{icon} {message}"

        elif self._error:
            if self._colors_supported:
                icon = self._colorize("✗", Colors.RED)
                message = self._colorize(
                    self._error_message or self.config.error_message, Colors.RED
                )
            else:
                icon = "✗"
                message = self._error_message or self.config.error_message

            final_line = f"{icon} {message}"

        else:
            # Stopped without success/error
            final_line = ""

        if final_line:
            self._write_line(final_line + "\n")
        else:
            # Just clear the line
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def stop(self, success: bool = True, error_message: str = ""):
        """
        Stop the spinner.

        Args:
            success: Whether the operation was successful
            error_message: Optional error message
        """
        self._success = success and not error_message
        self._error = not success or bool(error_message)
        self._error_message = error_message
        self._stop_event.set()

    def update_message(self, message: str):
        """
        Update the spinner message.

        Args:
            message: New message to display
        """
        self._current_message = message

    def is_running(self) -> bool:
        """Check if the spinner is running."""
        return self.is_alive() and not self._stop_event.is_set()


class Spinner:
    """
    Context manager for easy spinner usage.

    This class provides a convenient context manager interface
    for using the spinner in async operations.
    """

    def __init__(
        self,
        message: str = "Processing...",
        style: SpinnerStyle = SpinnerStyle.DOTS,
        success_message: str = "Done!",
        error_message: str = "Error!",
        color_enabled: bool = True,
    ):
        """
        Initialize the spinner context manager.

        Args:
            message: Message to display
            style: Spinner style
            success_message: Success message
            error_message: Error message
            color_enabled: Whether to enable colors
        """
        self.config = SpinnerConfig(
            style=style,
            message=message,
            success_message=success_message,
            error_message=error_message,
            color_enabled=color_enabled,
        )
        self.thread: Optional[SpinnerThread] = None

    def __enter__(self):
        """Enter the context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if exc_type is not None:
            # Exception occurred
            error_message = str(exc_val) if exc_val else "An error occurred"
            self.stop(success=False, error_message=error_message)
        else:
            # Success
            self.stop(success=True)

    def start(self):
        """Start the spinner."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = SpinnerThread(self.config)
            self.thread.start()

    def stop(self, success: bool = True, error_message: str = ""):
        """
        Stop the spinner.

        Args:
            success: Whether the operation was successful
            error_message: Optional error message
        """
        if self.thread and self.thread.is_alive():
            self.thread.stop(success=success, error_message=error_message)
            self.thread.join(timeout=1.0)  # Wait up to 1 second for cleanup

    def update_message(self, message: str):
        """
        Update the spinner message.

        Args:
            message: New message to display
        """
        if self.thread and self.thread.is_alive():
            self.thread.update_message(message)

    def is_running(self) -> bool:
        """Check if the spinner is running."""
        return self.thread is not None and self.thread.is_running()


def create_spinner(
    message: str = "Processing...",
    style: str = "dots",
    success_message: str = "Done!",
    error_message: str = "Error!",
    color_enabled: bool = True,
) -> Spinner:
    """
    Create a spinner with the specified configuration.

    Args:
        message: Message to display
        style: Spinner style name
        success_message: Success message
        error_message: Error message
        color_enabled: Whether to enable colors

    Returns:
        Configured spinner instance
    """
    # Convert string style to enum
    try:
        spinner_style = SpinnerStyle(style.lower())
    except ValueError:
        spinner_style = SpinnerStyle.DOTS

    return Spinner(
        message=message,
        style=spinner_style,
        success_message=success_message,
        error_message=error_message,
        color_enabled=color_enabled,
    )


# Convenience functions
def show_spinner(
    message: str = "Processing...", style: str = "dots", color_enabled: bool = True
) -> Spinner:
    """
    Show a spinner with the specified message.

    Args:
        message: Message to display
        style: Spinner style name
        color_enabled: Whether to enable colors

    Returns:
        Running spinner instance
    """
    spinner = create_spinner(message=message, style=style, color_enabled=color_enabled)
    spinner.start()
    return spinner


def print_success(message: str, color_enabled: bool = True):
    """
    Print a success message with checkmark.

    Args:
        message: Success message
        color_enabled: Whether to enable colors
    """
    if color_enabled and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        icon = f"{Colors.GREEN}✓{Colors.RESET}"
        text = f"{Colors.GREEN}{message}{Colors.RESET}"
    else:
        icon = "✓"
        text = message

    print(f"{icon} {text}")


def print_error(message: str, color_enabled: bool = True):
    """
    Print an error message with X mark.

    Args:
        message: Error message
        color_enabled: Whether to enable colors
    """
    if color_enabled and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        icon = f"{Colors.RED}✗{Colors.RESET}"
        text = f"{Colors.RED}{message}{Colors.RESET}"
    else:
        icon = "✗"
        text = message

    print(f"{icon} {text}")


def print_warning(message: str, color_enabled: bool = True):
    """
    Print a warning message with warning icon.

    Args:
        message: Warning message
        color_enabled: Whether to enable colors
    """
    if color_enabled and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        icon = f"{Colors.YELLOW}⚠{Colors.RESET}"
        text = f"{Colors.YELLOW}{message}{Colors.RESET}"
    else:
        icon = "⚠"
        text = message

    print(f"{icon} {text}")


def print_info(message: str, color_enabled: bool = True):
    """
    Print an info message with info icon.

    Args:
        message: Info message
        color_enabled: Whether to enable colors
    """
    if color_enabled and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        icon = f"{Colors.BLUE}ℹ{Colors.RESET}"
        text = f"{Colors.BLUE}{message}{Colors.RESET}"
    else:
        icon = "ℹ"
        text = message

    print(f"{icon} {text}")


# Export commonly used classes and functions
__all__ = [
    "Spinner",
    "SpinnerThread",
    "SpinnerStyle",
    "SpinnerConfig",
    "Colors",
    "create_spinner",
    "show_spinner",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
]
