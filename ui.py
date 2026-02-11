"""
Rich TUI layer for CLIAI.

Handles all terminal rendering: welcome banner, input prompts,
streaming markdown output, spinners, and status displays.
"""

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML
from typing import Optional, Generator

from config import Config, DATA_DIR

# Persistent input history file
HISTORY_FILE = DATA_DIR / "input_history"

console = Console()


class ChatUI:
    """Rich terminal interface for the chat session."""

    def __init__(self, config: Config):
        self.config = config
        self._session = PromptSession(
            history=FileHistory(str(HISTORY_FILE))
        )

    # ── Banner & Info ───────────────────────────────────────────

    def show_welcome(self):
        """Display the welcome banner with config info."""
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column("Key", style="dim")
        info_table.add_column("Value", style="bold")
        info_table.add_row("Endpoint", self.config.display_endpoint())
        info_table.add_row("Model", self.config.model)
        info_table.add_row("Profile", self.config.profile_name)
        if self.config.api_key:
            info_table.add_row("API Key", self.config.display_key())
        info_table.add_row("Streaming", "✓" if self.config.stream else "✗")
        if self.config.enforce_allowlist:
            hosts = ", ".join(self.config.allowed_hosts) if self.config.allowed_hosts else "(none)"
            info_table.add_row("🔒 Allowlist", hosts)

        panel = Panel(
            info_table,
            title="[bold bright_cyan]⚡ CLIAI[/bold bright_cyan]",
            subtitle="[dim]Type /help for commands • /exit to quit[/dim]",
            border_style="bright_cyan",
            padding=(1, 2),
        )
        console.print()
        console.print(panel)
        console.print()

    def show_info(self, config: Config):
        """Display current configuration info."""
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column("Key", style="dim")
        info_table.add_column("Value", style="bold")
        for key, val in config.to_dict().items():
            if key == "profile_name":
                continue
            info_table.add_row(key, str(val))

        panel = Panel(
            info_table,
            title=f"[bold]Profile: {config.profile_name}[/bold]",
            border_style="bright_cyan",
        )
        console.print(panel)

    # ── Input ───────────────────────────────────────────────────

    def get_input(self) -> Optional[str]:
        """
        Get multi-line user input.
        End a line with \\ to continue on next line.
        Returns None on EOF.
        """
        try:
            user_input = ""
            while True:
                if not user_input:
                    prompt_text = HTML(
                        '<style fg="ansibrightcyan" bg="" bold="true">You › </style>'
                    )
                else:
                    prompt_text = HTML(
                        '<style fg="ansigray">... </style>'
                    )

                line = self._session.prompt(prompt_text)

                if line.strip().endswith("\\"):
                    user_input += line.rstrip()[:-1] + "\n"
                else:
                    user_input += line
                    break

            return user_input
        except KeyboardInterrupt:
            return ""  # Empty string = skip this input
        except EOFError:
            return None  # None = exit

    # ── Output ──────────────────────────────────────────────────

    def stream_response(self, chunks: Generator) -> tuple[str, Optional[dict]]:
        """
        Render streaming chunks as live-updating markdown.

        Args:
            chunks: Generator yielding ChatChunk objects.

        Returns:
            Tuple of (full_response_text, usage_dict_or_None).
        """
        from client import ChatChunk

        full_response = ""
        usage = None
        interrupted = False

        console.print()
        console.print(
            Text("AI ›", style="bold bright_magenta"), end=" "
        )

        try:
            # Show spinner until first content arrives
            first_chunk = True
            with console.status(
                "[dim]Thinking…[/dim]", spinner="dots", spinner_style="bright_magenta"
            ) as status:
                for chunk in chunks:
                    if chunk.error:
                        status.stop()
                        self.show_error(chunk.error)
                        return "", None

                    if chunk.usage:
                        usage = chunk.usage

                    if chunk.delta_content:
                        if first_chunk:
                            status.stop()
                            first_chunk = False
                        break

                    if chunk.finish_reason:
                        status.stop()
                        break
                else:
                    # Generator was empty
                    return "", usage

            # Now stream remaining content with live markdown display
            if chunk.delta_content:
                full_response = chunk.delta_content

            with Live(
                Markdown(full_response),
                console=console,
                refresh_per_second=12,
                vertical_overflow="visible",
            ) as live:
                try:
                    for chunk in chunks:
                        if chunk.error:
                            live.stop()
                            self.show_error(chunk.error)
                            return full_response, usage

                        if chunk.usage:
                            usage = chunk.usage

                        if chunk.delta_content:
                            full_response += chunk.delta_content
                            live.update(Markdown(full_response))

                        if chunk.finish_reason == "interrupted":
                            interrupted = True
                            break

                        if chunk.finish_reason:
                            break
                except KeyboardInterrupt:
                    interrupted = True

        except KeyboardInterrupt:
            interrupted = True

        if interrupted:
            console.print("\n[bold yellow]⚠ Response stopped.[/bold yellow]")

        console.print()

        # Show usage if available
        if usage:
            self._show_usage(usage)

        return full_response, usage

    def show_non_stream_response(self, content: str, usage: Optional[dict] = None):
        """Display a non-streamed response."""
        console.print()
        console.print(Text("AI ›", style="bold bright_magenta"))
        console.print(Markdown(content))
        console.print()

        if usage:
            self._show_usage(usage)

    def _show_usage(self, usage: dict):
        """Display token usage info."""
        prompt_tokens = usage.get("prompt_tokens", "?")
        completion_tokens = usage.get("completion_tokens", "?")
        total_tokens = usage.get("total_tokens", "?")
        console.print(
            f"  [dim]tokens: {prompt_tokens} prompt + "
            f"{completion_tokens} completion = {total_tokens} total[/dim]"
        )

    # ── Status Messages ─────────────────────────────────────────

    def show_error(self, message: str):
        """Display an error message."""
        console.print(f"\n[bold red]✗ Error:[/bold red] {message}")

    def show_warning(self, message: str):
        """Display a warning message."""
        console.print(f"[bold yellow]⚠ {message}[/bold yellow]")

    def show_success(self, message: str):
        """Display a success message."""
        console.print(f"[bold green]✓ {message}[/bold green]")

    def show_info_msg(self, message: str):
        """Display an info message."""
        console.print(f"[dim]{message}[/dim]")

    def show_goodbye(self):
        """Display goodbye message."""
        console.print("\n[bold bright_cyan]👋 Goodbye![/bold bright_cyan]\n")

    def show_help(self, commands: dict[str, str]):
        """Display the help table."""
        table = Table(
            title="Available Commands",
            border_style="bright_cyan",
            show_lines=False,
        )
        table.add_column("Command", style="bold bright_cyan")
        table.add_column("Description")

        for cmd, desc in commands.items():
            table.add_row(cmd, desc)

        console.print(table)

    # ── Redaction Review ────────────────────────────────────────

    def redaction_review(
        self, original_text: str, redacted_text: str, redactions: list
    ) -> tuple[str, str, list, bool]:
        """
        Interactive redaction confirmation gate.

        Shows the redacted version of the user's input and allows them to
        send, add manual redactions, unredact items, or cancel.

        Args:
            original_text: The original user input.
            redacted_text: The text after auto-redaction.
            redactions: List of Redaction objects applied.

        Returns:
            (final_original, final_redacted, final_redactions, should_send)
        """
        from redactor import Redactor

        current_redacted = redacted_text
        current_redactions = list(redactions)

        while True:
            # Build the review panel
            self._show_redaction_panel(current_redacted, current_redactions)

            # Prompt for action
            try:
                action = self._session.prompt(
                    HTML(
                        '<style fg="ansiyellow" bold="true">'
                        'send / [a]dd / [u]nredact # / [c]ancel › '
                        '</style>'
                    )
                ).strip().lower()
            except (KeyboardInterrupt, EOFError):
                return original_text, current_redacted, current_redactions, False

            if action == "send":
                return original_text, current_redacted, current_redactions, True

            elif action == "c" or action == "cancel":
                return original_text, current_redacted, current_redactions, False

            elif action == "a" or action.startswith("add"):
                # Prompt for text to redact
                try:
                    text_to_mask = self._session.prompt(
                        HTML('<style fg="ansiyellow">Text to redact › </style>')
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    continue

                if not text_to_mask:
                    self.show_warning("No text provided.")
                    continue

                if text_to_mask not in original_text and text_to_mask not in current_redacted:
                    self.show_warning(f"'{text_to_mask}' not found in input.")
                    continue

                # Use the redactor to add a manual redaction
                # We need a reference to the Redactor, so we accept it as a standalone
                # We'll handle this by creating a new Redaction inline
                from redactor import Redaction
                tag_num = len(current_redactions) + 1
                placeholder = f"[REDACTED_{tag_num}]"
                redaction = Redaction(text_to_mask, placeholder, "manual", auto=False)
                current_redacted = current_redacted.replace(text_to_mask, placeholder)
                current_redactions.append(redaction)
                self.show_success(f"Redacted: {text_to_mask} → {placeholder}")

            elif action.startswith("u"):
                # Parse the number: "u 1", "u1", "unredact 2"
                parts = action.split()
                num_str = ""
                if len(parts) == 1:
                    # "u1" or "u"
                    num_str = parts[0].lstrip("unredact").strip()
                elif len(parts) >= 2:
                    num_str = parts[-1]

                if not num_str.isdigit():
                    self.show_error("Usage: u <number>  (e.g., u 1)")
                    continue

                idx = int(num_str) - 1
                if idx < 0 or idx >= len(current_redactions):
                    self.show_error(
                        f"Invalid number. Choose 1-{len(current_redactions)}."
                    )
                    continue

                r = current_redactions[idx]
                current_redacted = current_redacted.replace(r.placeholder, r.original)
                current_redactions.pop(idx)
                self.show_success(f"Unredacted: {r.placeholder} → {r.original}")

            else:
                self.show_error("Unknown action. Use [s]end, [a]dd, [u]nredact #, or [c]ancel.")

    def _show_redaction_panel(self, redacted_text: str, redactions: list):
        """Display the redaction review panel."""
        from rich.columns import Columns

        # Build content
        content_parts = []
        content_parts.append(
            Text(redacted_text, style="white")
        )

        if redactions:
            content_parts.append(Text(""))  # spacing

            table = Table(
                show_header=True, box=None, padding=(0, 2),
                header_style="dim bold",
            )
            table.add_column("#", style="dim", width=3)
            table.add_column("Original", style="red strike")
            table.add_column("→", style="dim", width=1)
            table.add_column("Masked As", style="bold bright_green")
            table.add_column("Type", style="dim")

            for i, r in enumerate(redactions, 1):
                auto_tag = "" if r.auto else "manual"
                table.add_row(
                    str(i),
                    r.original,
                    "→",
                    r.placeholder,
                    r.category if r.auto else auto_tag,
                )

            content_parts.append(table)
        else:
            content_parts.append(Text(""))
            content_parts.append(
                Text("  No sensitive data detected.", style="dim italic")
            )

        from rich.console import Group
        panel = Panel(
            Group(*content_parts),
            title="[bold yellow]🔒 Redaction Review[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
        console.print()
        console.print(panel)

    def show_redaction_mapping(self, mapping: list[tuple[str, str, str]]):
        """Display the current session redaction mapping."""
        if not mapping:
            console.print("[dim]No redactions in this session yet.[/dim]")
            return

        table = Table(
            title="Active Redaction Mapping",
            border_style="yellow",
        )
        table.add_column("Original", style="red")
        table.add_column("Placeholder", style="bold bright_green")

        for original, placeholder, _ in mapping:
            table.add_row(original, placeholder)

        console.print(table)
