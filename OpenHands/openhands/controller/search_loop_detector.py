"""Detects and intervenes when agents get stuck in repetitive search loops.

Inspired by OpenClaw's tool-loop detection (genericRepeat, knownPollNoProgress,
pingPong detectors) and corrective system message injection.

This module tracks recent search commands and their outcomes. When it detects
repeated failures with the same or similar search targets, it injects corrective
hints into observations to nudge the agent toward a different strategy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger('openhands.controller.search_loop_detector')

# Regex patterns to extract the search target from common search commands
# e.g., grep -r "foo" . → "foo"
#        grep -rn 'bar' /path → "bar"
#        rg "baz" --type py → "baz"
_GREP_TARGET_RE = re.compile(
    r'''(?:grep|rg|ack)\b.*?(?:['"](.+?)['"]|(?:-[a-zA-Z]*\s+)*(\S+))''',
    re.DOTALL,
)

# More specific: match the quoted search target in grep/rg commands
_QUOTED_SEARCH_RE = re.compile(
    r'''(?:grep|rg|ack)\b[^"']*["']([^"']+)["']'''
)

# Detect find -exec grep pattern
_FIND_EXEC_GREP_RE = re.compile(
    r'''find\b.*-exec\s+grep\b[^"']*["']([^"']+)["']'''
)

# Detect bash syntax errors
SYNTAX_ERROR_NEEDLES = (
    'syntax error',
    'unexpected token',
    'Unmatched',
    'unterminated',
)

# Detect empty search output (grep/rg with no results)
EMPTY_SEARCH_INDICATORS = (
    'grep:',  # grep error output starts with "grep:"
)


@dataclass
class SearchAttempt:
    """A record of a search command and its outcome."""
    command: str
    search_target: str
    was_empty: bool
    was_syntax_error: bool


@dataclass
class SearchLoopDetector:
    """Tracks recent search commands to detect stuck loops.

    Maintains a sliding window of recent search attempts and detects:
    1. Repeated searches for the same unverified target (A2 pattern)
    2. Repeated bash syntax errors with similar commands
    3. Consecutive stall actions (think/recall) without concrete progress (C1 pattern)
    """

    history_size: int = 20
    repeat_threshold: int = 2
    stall_threshold: int = 3

    _search_history: list[SearchAttempt] = field(default_factory=list)
    _consecutive_stalls: int = 0
    _last_was_concrete_action: bool = False

    def extract_search_target(self, command: str) -> str | None:
        """Extract the primary search target from a grep/rg/find command.

        Returns None if the command is not a search command.
        """
        m = _FIND_EXEC_GREP_RE.search(command)
        if m:
            return m.group(1)

        m = _QUOTED_SEARCH_RE.search(command)
        if m:
            return m.group(1)

        return None

    def is_search_command(self, command: str) -> bool:
        """Check if a command is a search-type command."""
        cmd_lower = command.strip().lower()
        return any(
            cmd_lower.startswith(prefix)
            for prefix in ('grep', 'rg', 'ack', 'find')
        ) or bool(re.search(r'\bgrep\b|\brg\b', cmd_lower))

    def record_search_result(
        self, command: str, output: str, exit_code: int | None = None
    ) -> str | None:
        """Record a search result and return a recovery hint if a loop is detected.

        Returns:
            A recovery hint string to append to the observation, or None.
        """
        if not self.is_search_command(command):
            return None

        search_target = self.extract_search_target(command)
        if not search_target:
            return None

        was_syntax_error = any(
            needle in output for needle in SYNTAX_ERROR_NEEDLES
        )
        was_empty = (
            not was_syntax_error
            and len(output.strip()) == 0
            and exit_code != 0
        )
        # Also treat grep returning only error lines as empty
        if not was_syntax_error and not was_empty:
            lines = output.strip().split('\n')
            if all(
                line.startswith('grep:') or line.strip() == ''
                for line in lines
            ) and len(lines) <= 3:
                was_empty = True

        attempt = SearchAttempt(
            command=command,
            search_target=search_target,
            was_empty=was_empty,
            was_syntax_error=was_syntax_error,
        )
        self._search_history.append(attempt)

        if len(self._search_history) > self.history_size:
            self._search_history = self._search_history[-self.history_size:]

        if was_syntax_error:
            return self._check_syntax_error_loop(attempt)
        if was_empty:
            return self._check_empty_search_loop(attempt)
        return None

    def _check_empty_search_loop(self, current: SearchAttempt) -> str | None:
        """Check if the same search target has failed repeatedly."""
        target_lower = current.search_target.lower()
        consecutive_empty = 0

        for attempt in reversed(self._search_history):
            if attempt.search_target.lower() == target_lower and attempt.was_empty:
                consecutive_empty += 1
            elif attempt.search_target.lower() == target_lower:
                break
            else:
                break

        if consecutive_empty >= self.repeat_threshold:
            logger.warning(
                f'Search loop detected: "{current.search_target}" returned empty '
                f'{consecutive_empty} consecutive times'
            )
            if consecutive_empty >= self.repeat_threshold + 1:
                return (
                    f'\n\n[CRITICAL: Repeated Search Failure — {consecutive_empty} times]\n'
                    f'You have searched for "{current.search_target}" {consecutive_empty} times with NO results. '
                    f'This target almost certainly does NOT exist in the codebase. '
                    f'You MUST abandon this search target completely. '
                    f'Instead: read the relevant file directly with `cat`, or search for a single broad keyword.'
                )
            return (
                f'\n\n[Empty Search Recovery Hint]\n'
                f'The search for "{current.search_target}" returned no results again. '
                f'Do NOT repeat a similar search for this target. '
                f'Instead: (1) verify the target exists by listing candidate files first (`find . -name "*.py" | head -20`), '
                f'(2) broaden the search to a single key token (e.g., just the class/function name without module prefix), '
                f'or (3) read the relevant file directly with `cat`.'
            )
        return None

    def _check_syntax_error_loop(self, current: SearchAttempt) -> str | None:
        """Check if similar commands keep producing syntax errors."""
        target_lower = current.search_target.lower()
        consecutive_errors = 0

        for attempt in reversed(self._search_history):
            if attempt.was_syntax_error and attempt.search_target.lower() == target_lower:
                consecutive_errors += 1
            else:
                break

        if consecutive_errors >= self.repeat_threshold:
            logger.warning(
                f'Syntax error loop detected: commands with target "{current.search_target}" '
                f'produced {consecutive_errors} consecutive syntax errors'
            )
            return (
                f'\n\n[Syntax Error Recovery Hint]\n'
                f'Your command had a syntax error for the {consecutive_errors}th time. '
                f'Do NOT retry the same command pattern. '
                f'Common fix: avoid `find -exec ... | grep ...` (pipes do not work inside -exec). '
                f'Instead use: `grep -rn "{current.search_target}" . --include="*.py"` '
                f'or `grep -rn "{current.search_target}" <specific_directory>`.'
            )
        return None

    def record_stall_action(self, action_type: str) -> str | None:
        """Record a stall action (think/recall/system/message) and return hint if threshold reached.

        Returns:
            A hint string if consecutive stalls exceed threshold, or None.
        """
        self._consecutive_stalls += 1

        if self._consecutive_stalls >= self.stall_threshold:
            count = self._consecutive_stalls
            logger.warning(
                f'Consecutive stall actions detected: {count} '
                f'(think/recall/system/message in a row)'
            )
            if count >= self.stall_threshold + 2:
                return (
                    f'[CRITICAL: {count} consecutive non-productive actions]\n'
                    f'You have used {count} consecutive think/recall steps without taking any concrete action. '
                    f'You MUST take a concrete action NOW: run a search command, read a file, or make an edit. '
                    f'Do NOT call think again until you have taken at least one concrete action.'
                )
            return (
                f'[Stall Warning: {count} consecutive non-productive actions]\n'
                f'You have called think/recall {count} times in a row without any concrete action '
                f'(search, read, edit). Each step counts toward your iteration limit. '
                f'Take a concrete action now — run a command, read a file, or make an edit.'
            )
        return None

    def record_concrete_action(self) -> None:
        """Record that a concrete action (run/edit/read) was taken. Resets stall counter."""
        self._consecutive_stalls = 0

    def get_repeat_count_for_target(self, search_target: str) -> int:
        """Get the number of consecutive empty searches for a target."""
        target_lower = search_target.lower()
        count = 0
        for attempt in reversed(self._search_history):
            if attempt.search_target.lower() == target_lower and (
                attempt.was_empty or attempt.was_syntax_error
            ):
                count += 1
            else:
                break
        return count
