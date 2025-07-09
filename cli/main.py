"""
CLI Main Entry Point.

This module provides the main CLI interface for the WhatsApp Sales Assistant,
allowing users to interact with the system through a command-line interface.
"""

import asyncio
import sys
import signal
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from .client import (
    OrchestratorClient,
    OrchestratorClientConfig,
    RequestStatus,
)
from .spinner import Spinner, SpinnerStyle, print_success, print_error, print_warning, print_info
from shared.utils import sanitize_text
from shared.observability import get_logger


# Configure logger
logger = get_logger(__name__)

# Create console for rich output
console = Console()

# Global client instance
client: Optional[OrchestratorClient] = None

# Session management
current_session_id: Optional[str] = None
current_user_id: str = "cli_user"
conversation_history: list = []


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print_info("\nShutting down CLI...")
    sys.exit(0)


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def initialize_client(
    base_url: str, timeout: float, retries: int, api_key: Optional[str] = None
) -> OrchestratorClient:
    """
    Initialize the orchestrator client.

    Args:
        base_url: Base URL of the orchestrator service
        timeout: Request timeout
        retries: Retry attempts
        api_key: Optional API key

    Returns:
        Configured orchestrator client
    """
    config = OrchestratorClientConfig(
        base_url=base_url, request_timeout=timeout, max_retries=retries, api_key=api_key
    )

    return OrchestratorClient(config)


async def test_connection(client: OrchestratorClient) -> bool:
    """
    Test connection to the orchestrator service.

    Args:
        client: Orchestrator client

    Returns:
        True if connection successful, False otherwise
    """
    with Spinner(
        message="Testing connection to orchestrator...",
        style=SpinnerStyle.DOTS,
        success_message="Connection successful!",
        error_message="Connection failed!",
    ):
        try:
            return await client.test_connection()
        except Exception as e:
            logger.error("Connection test failed", error=str(e))
            return False


async def send_message(client: OrchestratorClient, message: str) -> Optional[Dict[str, Any]]:
    """
    Send a message to the orchestrator.

    Args:
        client: Orchestrator client
        message: User message

    Returns:
        Response data or None if failed
    """
    global current_session_id, current_user_id, conversation_history

    # Initialize session if not exists
    if not current_session_id:
        current_session_id = f"cli_session_{int(time.time())}"

    # Sanitize message
    sanitized_message = sanitize_text(message, max_length=1000)

    with Spinner(
        message="Processing your message...",
        style=SpinnerStyle.DOTS,
        success_message="Message processed!",
        error_message="Failed to process message",
    ) as spinner:
        try:
            # Send message
            result = await client.send_message(
                user_message=sanitized_message,
                user_id=current_user_id,
                session_id=current_session_id,
            )

            if result.status == RequestStatus.SUCCESS:
                # Add to conversation history
                conversation_history.append(
                    {
                        "role": "user",
                        "content": sanitized_message,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                response_data = result.data
                if response_data:
                    conversation_history.append(
                        {
                            "role": "assistant",
                            "content": response_data.get("response", ""),
                            "timestamp": datetime.now().isoformat(),
                            "metadata": {
                                "classification": response_data.get("classification"),
                                "response_type": response_data.get("response_type"),
                                "processing_time": response_data.get("processing_time"),
                            },
                        }
                    )

                return response_data
            else:
                spinner.stop(success=False, error_message=result.error or "Unknown error")
                return None

        except Exception as e:
            logger.error("Failed to send message", error=str(e))
            spinner.stop(success=False, error_message=str(e))
            return None


def display_response(response_data: Dict[str, Any]):
    """
    Display the response in a formatted way.

    Args:
        response_data: Response data from orchestrator
    """
    # Extract response information
    response_text = response_data.get("response", "No response received")
    classification = response_data.get("classification")
    processing_time = response_data.get("processing_time")

    # Create response panel
    response_panel = Panel(
        response_text,
        title="🤖 Assistant Response",
        title_align="left",
        border_style="blue",
        padding=(1, 2),
    )

    console.print(response_panel)

    # Display classification information if available
    if classification:
        classification_info = []

        label = classification.get("label", "unknown")
        confidence = classification.get("confidence", 0.0)

        # Format classification with color
        if label == "product_information":
            label_color = "green"
            label_icon = "🛍️"
        elif label == "PQR":
            label_color = "yellow"
            label_icon = "❓"
        else:
            label_color = "red"
            label_icon = "❌"

        classification_info.append(f"{label_icon} [bold {label_color}]{label}[/bold {label_color}]")
        classification_info.append(f"📊 Confidence: {confidence:.2%}")

        if processing_time:
            classification_info.append(f"⏱️ Processing Time: {processing_time:.2f}s")

        # Display classification table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Info", style="dim")

        for info in classification_info:
            table.add_row(info)

        console.print(table)


def display_help():
    """Display help information."""
    help_text = """
[bold blue]WhatsApp Sales Assistant CLI[/bold blue]

[bold]Available Commands:[/bold]
• [cyan]help[/cyan] - Show this help message
• [cyan]status[/cyan] - Check service status
• [cyan]health[/cyan] - Check service health
• [cyan]metrics[/cyan] - Show service metrics
• [cyan]history[/cyan] - Show conversation history
• [cyan]clear[/cyan] - Clear conversation history
• [cyan]quit[/cyan] or [cyan]exit[/cyan] - Exit the CLI

[bold]Examples:[/bold]
• What's the price of iPhone 15?
• Do you have wireless headphones?
• My order is delayed
• I want to return this item

[bold]Tips:[/bold]
• Type naturally - the assistant will understand your intent
• Use [cyan]Ctrl+C[/cyan] to interrupt at any time
• The assistant classifies messages as product information or PQR (Problems/Queries/Complaints)
"""

    console.print(Panel(help_text, title="Help", border_style="green"))


def display_status(status_data: Dict[str, Any]):
    """
    Display service status information.

    Args:
        status_data: Status data from orchestrator
    """
    # Create status table
    table = Table(title="Service Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    # Add service information
    service_status = status_data.get("status", "unknown")
    service_color = "green" if service_status == "healthy" else "red"

    table.add_row(
        "Orchestrator Service",
        f"[{service_color}]{service_status}[/{service_color}]",
        f"Version: {status_data.get('version', 'unknown')}",
    )

    # Add health information
    health_data = status_data.get("health", {})
    classifier_status = health_data.get("classifier_connection", "unknown")
    classifier_color = "green" if classifier_status == "healthy" else "red"

    table.add_row(
        "Classifier Service",
        f"[{classifier_color}]{classifier_status}[/{classifier_color}]",
        f"Active Conversations: {health_data.get('active_conversations', 0)}",
    )

    # Add metrics
    metrics = status_data.get("metrics", {})
    table.add_row("Metrics", "📊 Active", f"Total Requests: {metrics.get('total_requests', 0)}")

    console.print(table)


def display_history():
    """Display conversation history."""
    if not conversation_history:
        print_info("No conversation history available.")
        return

    console.print(Panel("Conversation History", style="cyan"))

    for i, message in enumerate(conversation_history, 1):
        role = message["role"]
        content = message["content"]
        timestamp = message.get("timestamp", "")

        if role == "user":
            icon = "👤"
            style = "blue"
        else:
            icon = "🤖"
            style = "green"

        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_str = dt.strftime("%H:%M:%S")
        except Exception:
            time_str = ""

        console.print(f"{icon} [{style}]{role.title()}[/{style}] {time_str}")
        console.print(f"   {content}")

        # Show classification for assistant messages
        if role == "assistant":
            metadata = message.get("metadata", {})
            classification = metadata.get("classification")
            if classification:
                label = classification.get("label", "unknown")
                confidence = classification.get("confidence", 0.0)
                console.print(f"   [dim]Classification: {label} ({confidence:.2%})[/dim]")

        console.print()


async def interactive_mode(client: OrchestratorClient):
    """
    Run the interactive CLI mode.

    Args:
        client: Orchestrator client
    """
    global current_session_id, current_user_id

    # Welcome message
    welcome_text = f"""
[bold blue]WhatsApp Sales Assistant CLI[/bold blue]

Connected to: {client.config.base_url}
Session ID: {current_session_id}
User ID: {current_user_id}

Type [cyan]help[/cyan] for available commands, or start chatting!
Type [cyan]quit[/cyan] or [cyan]exit[/cyan] to leave.
"""

    console.print(Panel(welcome_text, title="Welcome", border_style="green"))

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold blue]You[/bold blue]", console=console).strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit"]:
                print_info("Goodbye!")
                break

            elif user_input.lower() == "help":
                display_help()
                continue

            elif user_input.lower() == "status":
                with Spinner("Getting service status..."):
                    result = await client.get_status()
                    if result.status == RequestStatus.SUCCESS:
                        display_status(result.data)
                    else:
                        print_error(f"Failed to get status: {result.error}")
                continue

            elif user_input.lower() == "health":
                with Spinner("Checking service health..."):
                    result = await client.get_health()
                    if result.status == RequestStatus.SUCCESS:
                        health_data = result.data
                        status = health_data.get("status", "unknown")
                        if status == "healthy":
                            print_success("Service is healthy!")
                        else:
                            print_warning(f"Service status: {status}")
                    else:
                        print_error(f"Health check failed: {result.error}")
                continue

            elif user_input.lower() == "metrics":
                with Spinner("Getting service metrics..."):
                    result = await client.get_metrics()
                    if result.status == RequestStatus.SUCCESS:
                        metrics = result.data
                        console.print(
                            Panel(
                                json.dumps(metrics, indent=2),
                                title="Service Metrics",
                                border_style="yellow",
                            )
                        )
                    else:
                        print_error(f"Failed to get metrics: {result.error}")
                continue

            elif user_input.lower() == "history":
                display_history()
                continue

            elif user_input.lower() == "clear":
                if Confirm.ask("Clear conversation history?"):
                    conversation_history.clear()
                    current_session_id = f"cli_session_{int(time.time())}"
                    print_success("Conversation history cleared!")
                continue

            # Send message to orchestrator
            response_data = await send_message(client, user_input)

            if response_data:
                display_response(response_data)
            else:
                print_error("Failed to get response from the assistant.")

        except KeyboardInterrupt:
            print_info("\nUse 'quit' or 'exit' to leave.")
            continue
        except Exception as e:
            logger.error("Error in interactive mode", error=str(e))
            print_error(f"An error occurred: {str(e)}")


@click.command()
@click.option("--url", default="http://localhost:8080", help="Orchestrator service URL")
@click.option("--timeout", default=30.0, type=float, help="Request timeout in seconds")
@click.option("--retries", default=3, type=int, help="Number of retry attempts")
@click.option("--api-key", help="API key for authentication")
@click.option("--user-id", default="cli_user", help="User ID for the session")
@click.option("--test-connection", is_flag=True, help="Test connection and exit")
@click.option("--message", "-m", help="Send a single message and exit")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(
    url: str,
    timeout: float,
    retries: int,
    api_key: Optional[str],
    user_id: str,
    test_connection: bool,
    message: Optional[str],
    verbose: bool,
):
    """
    WhatsApp Sales Assistant CLI.

    Interactive command-line interface for the multi-agent WhatsApp sales assistant.
    """
    global client, current_user_id, current_session_id

    # Set up signal handlers
    setup_signal_handlers()

    # Set user ID
    current_user_id = user_id
    current_session_id = f"cli_session_{int(time.time())}"

    # Initialize client
    client = initialize_client(url, timeout, retries, api_key)

    # Configure logging level
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    try:
        # Run async main
        asyncio.run(async_main(client, test_connection, message))
    except KeyboardInterrupt:
        print_info("\nGoodbye!")
    except Exception as e:
        logger.error("CLI failed", error=str(e))
        print_error(f"CLI failed: {str(e)}")
        sys.exit(1)


async def async_main(
    client: OrchestratorClient, test_connection_flag: bool, single_message: Optional[str]
):
    """
    Async main function.

    Args:
        client: Orchestrator client
        test_connection_flag: Whether to test connection only
        single_message: Optional single message to send
    """
    # Test connection
    if not await test_connection(client):
        print_error("Failed to connect to the orchestrator service.")
        print_info(f"Please ensure the service is running at {client.config.base_url}")
        sys.exit(1)

    if test_connection_flag:
        print_success("Connection test successful!")
        return

    # Send single message if provided
    if single_message:
        response_data = await send_message(client, single_message)
        if response_data:
            display_response(response_data)
        else:
            print_error("Failed to get response.")
            sys.exit(1)
        return

    # Run interactive mode
    await interactive_mode(client)


if __name__ == "__main__":
    main()
