#!/usr/bin/env python3
import os
import json
import re
import time
import threading
import argparse
import sys
import select
import asyncio

try:
    import termios
    import tty

    UNIX_TERMINAL = True
except ImportError:
    UNIX_TERMINAL = False
    termios = None
    tty = None
from collections import deque
from openai import AsyncOpenAI
from datasets import load_dataset
import tiktoken

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text
from rich.rule import Rule
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.style import Style
from rich import box

# -------------------------------------------------------------------------
# Global Configuration & Patterns
# -------------------------------------------------------------------------
_PATTERNS = (r"answer is \(?([A-J])\)?", r"[aA]nswer:\s*([A-J])", r"\b[A-J]\b")
API_KEY = ""
args = None
GLOBAL_TOKENIZER = None

console = Console()

# -------------------------------------------------------------------------
# Global Data Buffers (decoupled architecture)
# QUESTION_RUNNERS write to these, RENDERER reads from them
# -------------------------------------------------------------------------
QUESTION_STATE_BUFFER = {}
LOG_MESSAGES = []
LOG_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
RENDERING_ACTIVE = True


# -------------------------------------------------------------------------
# Buffer Access Functions (for question runners)
# -------------------------------------------------------------------------
def buffer_log(message):
    timestamp = time.strftime("%H:%M:%S")
    with LOG_LOCK:
        LOG_MESSAGES.append(f"[{timestamp}] {message}")
        if len(LOG_MESSAGES) > 50:
            LOG_MESSAGES.pop(0)


def buffer_update_question(q_num, **kwargs):
    with STATE_LOCK:
        if q_num not in QUESTION_STATE_BUFFER:
            QUESTION_STATE_BUFFER[q_num] = {
                "status": "pending",
                "tokens": 0,
                "rate": 0.0,
                "elapsed": 0,
                "stalled": False,
                "retry_count": 0,
            }
        QUESTION_STATE_BUFFER[q_num].update(kwargs)


def buffer_set_question_status(q_num, status):
    with STATE_LOCK:
        if q_num not in QUESTION_STATE_BUFFER:
            QUESTION_STATE_BUFFER[q_num] = {}
        QUESTION_STATE_BUFFER[q_num]["status"] = status


def buffer_get_state_snapshot():
    with STATE_LOCK:
        return dict(QUESTION_STATE_BUFFER)


def buffer_get_log_snapshot():
    with LOG_LOCK:
        return list(LOG_MESSAGES)


def buffer_clear():
    global QUESTION_STATE_BUFFER, LOG_MESSAGES
    with STATE_LOCK:
        QUESTION_STATE_BUFFER.clear()
    with LOG_LOCK:
        LOG_MESSAGES.clear()


# Braille spinner frames (feature 9)
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# FIX 2/3: Rolling window duration for tok/s calculation (seconds)
_RATE_WINDOW_SECS = 8.0
# FIX 4: Seconds without a token update before a question is considered stalled
_STALL_THRESHOLD_SECS = 25.0


def get_tokenizer():
    global GLOBAL_TOKENIZER
    if GLOBAL_TOKENIZER is None:
        GLOBAL_TOKENIZER = tiktoken.get_encoding("cl100k_base")
    return GLOBAL_TOKENIZER


# -------------------------------------------------------------------------
# File Lock & Shared State
# -------------------------------------------------------------------------
file_lock = threading.Lock()
GLOBAL_QUIT_REQUESTED = False
SCROLL_OFFSET = 0
INPUT_PROMPT_ACTIVE = False
AUTO_SCROLL = True  # auto-follow active questions; disabled on manual scroll


# -------------------------------------------------------------------------
# API Client
# -------------------------------------------------------------------------
def get_async_client():
    return AsyncOpenAI(api_key=API_KEY, base_url=args.url, timeout=3600.0)


# -------------------------------------------------------------------------
# Data Loading & Preprocessing
# -------------------------------------------------------------------------
def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    return preprocess(test_df), preprocess(val_df)


def preprocess(raw):
    cleaned = []
    for each in raw:
        opts = [opt for opt in each["options"] if opt != "N/A"]
        each["options"] = opts
        cleaned.append(each)
    grouped = {}
    for each in cleaned:
        grouped.setdefault(each["category"], []).append(each)
    return grouped


def format_example(question, options, cot_content=""):
    if not cot_content:
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = f"Question: {question}\nOptions: "
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += f"{choice_map[i]}. {opt}\n"
    if cot_content:
        example += f"Answer: {cot_content}\n\n"
    else:
        example += "Answer: "
    return example


def extract_answer(text):
    for pattern in _PATTERNS:
        if matches := re.findall(pattern, text):
            return matches[-1]
    return None


# -------------------------------------------------------------------------
# Exceptions
# -------------------------------------------------------------------------
class MaxTokensExceeded(Exception):
    pass


class MissingUsageError(Exception):
    pass


# -------------------------------------------------------------------------
# Active Question Tracker (Display & State)
# -------------------------------------------------------------------------
class ActiveQuestionTracker:
    def __init__(self, subject, total_questions, max_concurrent=4, loop=None):
        self.subject = subject
        self.total_questions = total_questions
        self.max_concurrent = max_concurrent
        self.completed = 0
        self.start_time = time.time()
        self.loop = loop
        self.active_questions = {}  # q_num -> (task, start_time, stop_event, restart, kill, retry_count)
        self.token_counts = {}
        # FIX 2: Replace single last_token_time + EMA rate with a rolling-window deque per question.
        # Each deque holds (timestamp, cumulative_token_count) samples.
        self.token_history = {}  # q_num -> deque of (ts, cum_tokens)
        self.token_rates = {}  # q_num -> computed tok/s (float)
        self.stalled = {}
        self.lock = threading.Lock()
        self.keyboard_listener_active = True

        # Wall State
        self.question_results = {}
        self.log_messages = []
        self.last_render_time = 0
        self.render_interval = (
            1.0  # Reduced to 1 second to avoid interfering with input
        )

        # Cache terminal size
        self.last_cols = 0
        self.last_rows = 0

        # Rich Live reference (set externally before first render)
        self.live = None

        # Animation state (features 9, 10)
        self._spinner_idx = 0
        self._stall_pulse = False  # flips each render tick

        self.log(f"Starting {subject} ({total_questions} questions)")
        self.log(
            f"Controls: [Up/Down] Scroll (disables auto-scroll) | [r] restart | [k] kill | [q] quit"
        )
        if args.retry > 0:
            self.log(f"Max token retries: {args.retry}")

    # -----------------------------------------------------------------------
    # State Mutators
    # -----------------------------------------------------------------------
    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {message}")
        if len(self.log_messages) > 50:
            self.log_messages.pop(0)
        buffer_log(message)

    def start_question(self, question_num, task, stop_event):
        with self.lock:
            self.active_questions[question_num] = (
                task,
                time.time(),
                stop_event,
                False,
                False,
                0,
            )
            self.token_counts[question_num] = 0
            self.token_history[question_num] = deque()
            self.token_rates[question_num] = 0.0
            self.stalled[question_num] = False
            self.question_results[question_num] = "active"

        buffer_update_question(
            question_num,
            status="active",
            tokens=0,
            rate=0.0,
            stalled=False,
            start_time=time.time(),
        )

    # FIX 2 & 3: `count_only` parameter prevents rate recalculation on the final
    # usage_data flush (which would cause a spike: large token jump, near-zero dt).
    # FIX 5: Resetting to 0 now also clears the history deque and rate.
    def update_token(self, question_num, token_count, count_only=False):
        rate = 0.0
        with self.lock:
            if question_num not in self.token_counts:
                return
            now = time.time()

            if count_only or token_count == 0:
                self.token_counts[question_num] = token_count
                if token_count == 0:
                    self.token_history[question_num] = deque()
                    self.token_rates[question_num] = 0.0
                self.stalled[question_num] = False
                rate = self.token_rates.get(question_num, 0.0)
                buffer_update_question(
                    question_num, tokens=token_count, rate=rate, stalled=False
                )
                return

            self.token_counts[question_num] = token_count

            hist = self.token_history[question_num]
            hist.append((now, token_count))
            cutoff = now - _RATE_WINDOW_SECS
            while hist and hist[0][0] < cutoff:
                hist.popleft()

            if len(hist) >= 2:
                dt = hist[-1][0] - hist[0][0]
                delta = hist[-1][1] - hist[0][1]
                if dt > 0.05 and delta > 0:
                    self.token_rates[question_num] = delta / dt
            rate = self.token_rates.get(question_num, 0.0)

            self.stalled[question_num] = False

        buffer_update_question(
            question_num, tokens=token_count, rate=rate, stalled=False
        )

    def set_stalled(self, question_num, stalled):
        with self.lock:
            if question_num in self.stalled:
                self.stalled[question_num] = stalled
        buffer_update_question(question_num, stalled=stalled)

    def complete_question(self, question_num, token_count=None, success=True):
        with self.lock:
            if question_num in self.active_questions:
                del self.active_questions[question_num]
                self.token_counts.pop(question_num, None)
                self.token_history.pop(question_num, None)
                self.token_rates.pop(question_num, None)
                self.stalled.pop(question_num, None)

            status = "success" if success else "wrong"
            self.question_results[question_num] = status
            if success:
                self.completed += 1

        buffer_set_question_status(question_num, status)

        if success:
            self.log(f"Q{question_num} completed ({token_count} tokens)")
        else:
            self.log(f"Q{question_num} failed/wrong answer")

    def kill_question(self, question_num):
        with self.lock:
            if question_num in self.active_questions:
                task, start_time, stop_event, _, _, retry_count = self.active_questions[
                    question_num
                ]
                self.active_questions[question_num] = (
                    task,
                    start_time,
                    stop_event,
                    False,
                    True,
                    retry_count,
                )
                if stop_event:
                    stop_event.set()
                self.log(f"Q{question_num} marked for kill")
                return True
            else:
                self.log(f"Q{question_num} is not active")
                return False

    def restart_question(self, question_num):
        with self.lock:
            if question_num in self.active_questions:
                task, start_time, stop_event, _, _, retry_count = self.active_questions[
                    question_num
                ]
                self.active_questions[question_num] = (
                    task,
                    start_time,
                    stop_event,
                    True,
                    False,
                    retry_count,
                )
                if stop_event:
                    stop_event.set()
                self.log(f"Q{question_num} marked for restart")
                return True
            else:
                self.log(f"Q{question_num} is not active")
                return False

    def increment_retry_count(self, question_num):
        with self.lock:
            if question_num in self.active_questions:
                task, start_time, stop_event, restart_flag, kill_flag, retry_count = (
                    self.active_questions[question_num]
                )
                retry_count += 1
                self.active_questions[question_num] = (
                    task,
                    start_time,
                    stop_event,
                    restart_flag,
                    kill_flag,
                    retry_count,
                )
                return retry_count
        return 0

    def get_retry_count(self, question_num):
        with self.lock:
            if question_num in self.active_questions:
                _, _, _, _, _, retry_count = self.active_questions[question_num]
                return retry_count
        return 0

    def get_flags(self, question_num):
        with self.lock:
            if question_num in self.active_questions:
                _, _, _, restart_flag, kill_flag, _ = self.active_questions[
                    question_num
                ]
                return restart_flag, kill_flag
            return False, False

    def clear_flags(self, question_num):
        with self.lock:
            if question_num in self.active_questions:
                task, start_time, stop_event, _, _, retry_count = self.active_questions[
                    question_num
                ]
                self.active_questions[question_num] = (
                    task,
                    start_time,
                    stop_event,
                    False,
                    False,
                    retry_count,
                )

    def stop_all_questions(self):
        with self.lock:
            for question_num, (task, start_time, stop_event, _, _, _) in list(
                self.active_questions.items()
            ):
                if stop_event:
                    stop_event.set()
                if task and not task.done() and self.loop:
                    self.loop.call_soon_threadsafe(task.cancel)
                self.token_counts.pop(question_num, None)
                self.token_history.pop(question_num, None)
                self.token_rates.pop(question_num, None)
                self.stalled.pop(question_num, None)
            self.active_questions.clear()
            self.keyboard_listener_active = False
        buffer_clear()

    def any_stalled(self):
        with self.lock:
            return any(self.stalled.values())

    def calculate_accuracy(self):
        finished = [
            v for v in self.question_results.values() if v in ("success", "wrong")
        ]
        total = len(finished)
        if total == 0:
            return 0.0
        return sum(1 for v in finished if v == "success") / total * 100

    # -----------------------------------------------------------------------
    # Feature 8: age-based colour for active question cells
    # -----------------------------------------------------------------------
    def _age_style(self, elapsed_secs: float) -> str:
        if elapsed_secs < 30:
            return "white"
        elif elapsed_secs < 90:
            return "yellow"
        elif elapsed_secs < 180:
            return "color(208)"  # orange
        else:
            return "red"

    # -----------------------------------------------------------------------
    # FIX 4: Stall detection — called once per render tick so no extra thread needed.
    # -----------------------------------------------------------------------
    def _check_stalls(self, now: float):
        """Mark questions stalled if their token history hasn't advanced recently."""
        with self.lock:
            for q_num, hist in self.token_history.items():
                if not hist:
                    # No samples yet (just started) — not stalled
                    continue
                last_ts = hist[-1][0]
                if now - last_ts > _STALL_THRESHOLD_SECS:
                    self.stalled[q_num] = True

    # -----------------------------------------------------------------------
    # Rich Rendering
    # Feature 1 : two-column Layout (grid left, detail+log right)
    # Feature 2 : active-questions detail Table  (right-top panel)
    # Feature 3 : subject header wrapped in a Panel with border title
    # Features 8,9,10 applied inside the grid and the detail table
    # -----------------------------------------------------------------------
    def get_renderable(self):
        now = time.time()

        # FIX 4: Detect stalls on every render tick (cheap, lock-free from caller's pov)
        self._check_stalls(now)

        try:
            term_size = os.get_terminal_size()
            cols = term_size.columns
            rows = term_size.lines
            self.last_cols, self.last_rows = cols, rows
        except OSError:
            cols = self.last_cols or 80
            rows = self.last_rows or 24

        # Advance animation state once per render tick
        self._spinner_idx += 1
        self._stall_pulse = not self._stall_pulse
        spinner_char = SPINNER_FRAMES[self._spinner_idx % len(SPINNER_FRAMES)]

        # Thread-safe snapshot - read from global buffer
        q_results = dict(self.question_results)
        active_qs = dict(self.active_questions)

        # Read active question state from global buffer
        buffer_state = buffer_get_state_snapshot()
        tok_counts = {q: s.get("tokens", 0) for q, s in buffer_state.items()}
        tok_rates = {q: s.get("rate", 0.0) for q, s in buffer_state.items()}
        stalled = {q: s.get("stalled", False) for q, s in buffer_state.items()}

        log_msgs = buffer_get_log_snapshot()

        correct = sum(1 for v in q_results.values() if v == "success")
        wrong_count = sum(1 for v in q_results.values() if v == "wrong")
        acc = self.calculate_accuracy()
        elapsed = int(now - self.start_time)
        total_rate = sum(tok_rates.values())
        total_toks = sum(tok_counts.values())

        # ----------------------------------------------------------------
        # Feature 3: Header Panel
        # ----------------------------------------------------------------
        hdr = Text(justify="left")
        hdr.append("Progress ", style="dim")
        hdr.append(f"{correct}", style="bold green")
        hdr.append(f"/{self.total_questions}   ", style="dim")
        hdr.append("Acc ", style="dim")
        hdr.append(f"{acc:.1f}%   ", style="bold yellow")
        hdr.append("Wrong ", style="dim")
        hdr.append(f"{wrong_count}   ", style="bold color(208)")
        hdr.append("Elapsed ", style="dim")
        hdr.append(f"{elapsed}s   ", style="dim")
        hdr.append("Tokens ", style="dim")
        hdr.append(f"{total_toks:,}   ", style="bold magenta")
        hdr.append("∑tok/s ", style="dim")
        hdr.append(f"{total_rate:.0f}", style="bold cyan")

        header_panel = Panel(
            hdr,
            title=f"[bold white]MMLU-Pro[/bold white]  [bold cyan]{self.subject}[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )

        # ----------------------------------------------------------------
        # Feature 2: Active Questions Detail Table (right-top)
        # ----------------------------------------------------------------
        active_table = Table(
            show_header=True,
            header_style="bold dim",
            box=box.SIMPLE,
            expand=True,
            show_edge=False,
            padding=(0, 1),
        )
        active_table.add_column("Q#", width=6, no_wrap=True)
        active_table.add_column("Time", width=6, no_wrap=True)
        active_table.add_column("Tokens", width=7, no_wrap=True)
        active_table.add_column("Tok/s", width=6, no_wrap=True)
        active_table.add_column("Retry", width=5, no_wrap=True)
        active_table.add_column("", width=2, no_wrap=True)  # spinner / stall indicator

        for q_num in sorted(active_qs.keys()):
            _, q_start, _, _, _, retry_cnt = active_qs[q_num]
            q_elapsed = now - q_start
            tokens = tok_counts.get(q_num, 0)
            rate = tok_rates.get(q_num, 0.0)
            is_stalled = stalled.get(q_num, False)
            age_sty = self._age_style(q_elapsed)

            if is_stalled:
                # Feature 10: pulse row style between bold-red and dim-red
                row_style = (
                    Style(bold=True, color="red")
                    if self._stall_pulse
                    else Style(color="dark_red")
                )
                indicator = Text("!", style="bold red")
            else:
                row_style = Style(color=age_sty)
                indicator = Text(spinner_char, style="green")  # Feature 9

            active_table.add_row(
                f"Q{q_num}",
                f"{int(q_elapsed)}s",
                str(tokens),
                f"{rate:.0f}" if rate > 0 else "—",
                str(retry_cnt) if retry_cnt > 0 else "—",
                indicator,
                style=row_style,
            )

        active_panel = Panel(
            active_table,
            title="[bold white]Active Questions[/bold white]",
            border_style="blue",
            padding=(0, 0),
        )

        # ----------------------------------------------------------------
        # Log panel (right-bottom)
        # ----------------------------------------------------------------
        log_text = Text()
        log_height = 10
        for msg in log_msgs[max(0, len(log_msgs) - log_height) :]:
            clean = "".join(c for c in msg if c.isprintable())
            log_text.append(clean + "\n", style="dim")

        log_panel = Panel(
            log_text,
            title="[bold white]Messages[/bold white]",
            border_style="blue",
            padding=(0, 1),
        )

        # ----------------------------------------------------------------
        # Question Grid (left panel)
        # Features 8 (age color), 9 (spinner), 10 (stall pulse) on cells
        # ----------------------------------------------------------------
        # Estimate inner width of the left (~3/4) panel
        left_inner_w = max(10, (cols * 3 // 4) - 4)
        block_w = 7  # chars per grid cell: [###]xx (no symbols, tight)
        grid_cols = max(1, left_inner_w // block_w)

        # rows used: header(3) + footer(1) + grid-panel borders(2) = 6
        content_height = max(1, rows - 6)
        total_rows = (self.total_questions + grid_cols - 1) // grid_cols

        global SCROLL_OFFSET, AUTO_SCROLL
        max_scroll = max(0, total_rows - content_height)
        SCROLL_OFFSET = max(0, min(SCROLL_OFFSET, max_scroll))

        # Auto-scroll: keep lowest active question row in view
        if AUTO_SCROLL and active_qs:
            lowest_active_row = max((q_num - 1) // grid_cols for q_num in active_qs)
            visible_end = SCROLL_OFFSET + content_height - 1
            if lowest_active_row > visible_end:
                SCROLL_OFFSET = min(max_scroll, lowest_active_row - content_height + 1)
            elif lowest_active_row < SCROLL_OFFSET:
                SCROLL_OFFSET = lowest_active_row

        start_idx = SCROLL_OFFSET * grid_cols
        end_idx = min(self.total_questions, start_idx + content_height * grid_cols)

        grid_text = Text()
        current_row_items = []

        for q_idx in range(start_idx, end_idx):
            q_num = q_idx + 1
            status = q_results.get(q_num, "pending")
            is_active = q_num in active_qs

            item = Text(no_wrap=True)

            if is_active:
                _, q_start, _, _, _, retry_cnt = active_qs[q_num]
                q_elapsed = now - q_start
                tokens = tok_counts.get(q_num, 0)
                is_stalled = stalled.get(q_num, False)
                age_sty = self._age_style(q_elapsed)  # Feature 8

                if is_stalled:
                    # Feature 10: alternate between bg highlight and plain red
                    if self._stall_pulse:
                        item.append(
                            f"{q_num:03d}",
                            style=Style(bold=True, color="white", bgcolor="red"),
                        )
                        item.append(
                            "!", style=Style(bold=True, color="white", bgcolor="red")
                        )
                    else:
                        item.append(f"{q_num:03d}", style=Style(bold=True, color="red"))
                        item.append("!", style=Style(bold=True, color="red"))
                else:
                    # q_num fixed white, spinner right of it, then token count age-colored
                    if tokens >= 10000:
                        tok_str = f"{tokens // 1000}"
                    elif tokens >= 1000:
                        tok_str = f"{tokens // 1000} "
                    else:
                        tok_str = f"{tokens:<3}"
                    item.append(f"{q_num:03d}", style="white")
                    item.append(spinner_char, style="green")
                    item.append(tok_str[:3], style=age_sty)

            elif status == "success":
                item.append(f"{q_num:03d}", style="green")
            elif status == "wrong":
                item.append(f"{q_num:03d}", style="bold color(208)")
            else:
                item.append(f"{q_num:03d}", style="bright_black")

            # Pad to fixed cell width
            pad = block_w - len(item.plain)
            if pad > 0:
                item.append(" " * pad)

            current_row_items.append(item)
            if len(current_row_items) == grid_cols:
                for it in current_row_items:
                    grid_text.append_text(it)
                grid_text.append("\n")
                current_row_items = []

        if current_row_items:
            for it in current_row_items:
                grid_text.append_text(it)
            grid_text.append("\n")

        if total_rows > content_height:
            auto_tag = "[auto]" if AUTO_SCROLL else "[manual - ↑↓ or 'a' to re-enable]"
            grid_text.append(
                f"── Scroll {SCROLL_OFFSET}/{max_scroll} (↑/↓) {auto_tag} ──",
                style="bright_black",
            )

        grid_panel = Panel(
            grid_text,
            title="[bold white]Question Wall[/bold white]",
            border_style="bright_black",
            padding=(0, 1),
        )

        # ----------------------------------------------------------------
        # Footer controls bar (one line, no panel)
        # ----------------------------------------------------------------
        controls = Text(justify="center")
        controls.append(" [r] ", style="bold yellow")
        controls.append("restart  ", style="dim")
        controls.append("[k] ", style="bold red")
        controls.append("kill  ", style="dim")
        controls.append("[↑↓] ", style="bold white")
        controls.append("scroll  ", style="dim")
        controls.append("[a] ", style="bold cyan")
        auto_lbl = "auto-scroll ON " if AUTO_SCROLL else "auto-scroll OFF"
        controls.append(auto_lbl + "  ", style="bold cyan" if AUTO_SCROLL else "dim")
        controls.append("[q] ", style="bold magenta")
        controls.append("quit  ", style="dim")
        controls.append(
            f"│  active {len(active_qs)}/{self.max_concurrent}", style="dim"
        )

        # ----------------------------------------------------------------
        # Feature 1: Two-column Layout assembly
        # ----------------------------------------------------------------
        layout = Layout()
        layout.split_column(
            Layout(header_panel, name="header", size=3),
            Layout(name="body"),
            Layout(controls, name="footer", size=1),
        )
        layout["body"].split_row(
            Layout(grid_panel, name="left", ratio=3),
            Layout(name="right", ratio=1),
        )
        layout["right"].split_column(
            Layout(active_panel, name="active", ratio=1),
            Layout(log_panel, name="log", ratio=1),
        )

        return layout

    def render_screen(self):
        """Throttled update of the Rich Live display."""
        global RENDERING_ACTIVE
        now = time.time()
        if not RENDERING_ACTIVE:
            return
        if now - self.last_render_time < self.render_interval:
            return
        self.last_render_time = now
        if self.live is not None:
            try:
                self.live.update(self.get_renderable(), refresh=True)
            except Exception:
                pass


# -------------------------------------------------------------------------
# Keyboard Listener
# FIX: SSH-compatible terminal handling - avoid switching between cbreak/canonical modes
# -------------------------------------------------------------------------
def keyboard_listener(tracker, event_loop):
    global \
        GLOBAL_QUIT_REQUESTED, \
        SCROLL_OFFSET, \
        INPUT_PROMPT_ACTIVE, \
        AUTO_SCROLL, \
        RENDERING_ACTIVE

    if not UNIX_TERMINAL:
        console.print(
            "\n[yellow]Terminal input not available on this platform - running in non-interactive mode[/yellow]"
        )
        console.print(
            "[dim]Note: Keyboard controls (r, k, q, arrow keys) are disabled.[/dim]"
        )
        while tracker.keyboard_listener_active and not GLOBAL_QUIT_REQUESTED:
            time.sleep(1)
        return

    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except termios.error:
        console.print(
            "\n[red]Terminal not available - running in non-interactive mode[/red]"
        )
        return

    input_buffer = []
    input_mode_active = False

    def read_key_nonblocking():
        """Read a single keypress, handling escape sequences for arrow keys."""
        try:
            if not select.select([fd], [], [], 0.01)[0]:
                return None
            raw = os.read(fd, 1)
            if not raw:
                return None
            key = raw.decode("utf-8", errors="ignore")

            if key == "\x1b":
                seq = ""
                for _ in range(4):
                    if select.select([fd], [], [], 0.05)[0]:
                        seq += os.read(fd, 1).decode("utf-8", errors="ignore")
                    else:
                        break
                if "[A" in seq:
                    return "UP"
                elif "[B" in seq:
                    return "DOWN"
                return "ESC"
            return key
        except Exception:
            return None

    def drain_input():
        """Clear any pending input from the buffer."""
        try:
            termios.tcflush(fd, termios.TCIFLUSH)
        except Exception:
            pass

    def restore_terminal_and_get_input(prompt_text):
        """Temporarily restore canonical mode for safe input over SSH."""
        global RENDERING_ACTIVE
        nonlocal input_mode_active

        RENDERING_ACTIVE = False

        try:
            termios.tcsetattr(fd, termios.TCSANOW, old_settings)
        except Exception:
            pass

        input_mode_active = False
        try:
            drain_input()
            sys.stdout.write("\r\n")
            sys.stdout.flush()
            result = input(prompt_text)
            sys.stdout.write("\r")
            sys.stdout.flush()
            return result
        finally:
            try:
                tty.setcbreak(fd)
                termios.tcflush(fd, termios.TCIFLUSH)
                input_mode_active = True
                RENDERING_ACTIVE = True
            except Exception:
                pass

    try:
        tty.setcbreak(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        tracker.log("Keyboard listener started (SSH-compatible mode).")

        while tracker.keyboard_listener_active and not GLOBAL_QUIT_REQUESTED:
            try:
                key = read_key_nonblocking()
                if key is None:
                    time.sleep(0.005)
                    continue

                if key == "UP":
                    AUTO_SCROLL = False
                    SCROLL_OFFSET = max(0, SCROLL_OFFSET - 1)
                    continue

                if key == "DOWN":
                    AUTO_SCROLL = False
                    SCROLL_OFFSET += 1
                    continue

                if key.lower() == "a":
                    AUTO_SCROLL = True
                    tracker.log("Auto-scroll re-enabled")
                    continue

                if key.lower() == "q":
                    INPUT_PROMPT_ACTIVE = True
                    live = tracker.live
                    if live is not None:
                        live.stop()

                    confirm = restore_terminal_and_get_input("Quit? (y/N): ")
                    confirm = (confirm or "").strip().lower()

                    if confirm == "y":
                        tracker.log("Quit requested.")
                        GLOBAL_QUIT_REQUESTED = True
                        tracker.stop_all_questions()
                        return

                    if live is not None:
                        live.start()
                    INPUT_PROMPT_ACTIVE = False
                    continue

                if key.lower() in ["r", "k"]:
                    INPUT_PROMPT_ACTIVE = True
                    with tracker.lock:
                        active_snap = dict(tracker.active_questions)
                        tok_snap = dict(tracker.token_counts)
                        rate_snap = dict(tracker.token_rates)
                        stalled_snap = dict(tracker.stalled)

                    if not active_snap:
                        tracker.log("No active questions")
                        time.sleep(1)
                        INPUT_PROMPT_ACTIVE = False
                        continue

                    live = tracker.live
                    if live is not None:
                        live.stop()

                    action = "restart" if key.lower() == "r" else "kill"
                    action_color = "yellow" if action == "restart" else "red"

                    pick_table = Table(
                        title=f"[bold {action_color}]Select question to {action}[/bold {action_color}]",
                        box=box.ROUNDED,
                        border_style=action_color,
                        header_style=f"bold {action_color}",
                        show_lines=False,
                        expand=False,
                    )
                    pick_table.add_column("Q#", width=6, no_wrap=True)
                    pick_table.add_column("Elapsed", width=8, no_wrap=True)
                    pick_table.add_column("Tokens", width=8, no_wrap=True)
                    pick_table.add_column("Tok/s", width=8, no_wrap=True)
                    pick_table.add_column("Retries", width=7, no_wrap=True)
                    pick_table.add_column("State", width=10, no_wrap=True)

                    now = time.time()
                    for q_num in sorted(active_snap.keys()):
                        _, q_start, _, _, _, retry_cnt = active_snap[q_num]
                        q_elapsed = now - q_start
                        tokens = tok_snap.get(q_num, 0)
                        rate = rate_snap.get(q_num, 0.0)
                        is_stalled = stalled_snap.get(q_num, False)

                        state_txt = (
                            Text("STALLED", style="bold red")
                            if is_stalled
                            else Text("active", style="green")
                        )

                        if q_elapsed < 30:
                            row_sty = "white"
                        elif q_elapsed < 90:
                            row_sty = "yellow"
                        elif q_elapsed < 180:
                            row_sty = "color(208)"
                        else:
                            row_sty = "red"

                        pick_table.add_row(
                            f"Q{q_num}",
                            f"{int(q_elapsed)}s",
                            str(tokens),
                            f"{rate:.0f}" if rate > 0 else "—",
                            str(retry_cnt) if retry_cnt > 0 else "—",
                            state_txt,
                            style=row_sty,
                        )

                    console.print(pick_table)
                    q_input = restore_terminal_and_get_input(
                        f"  Q number to {action} (Enter to cancel): "
                    )

                    if live is not None:
                        live.start()
                    INPUT_PROMPT_ACTIVE = False

                    if q_input:
                        try:
                            q_num = int(q_input.strip())
                            if key.lower() == "r":
                                if tracker.restart_question(q_num):
                                    tracker.log(f"Q{q_num} restart")
                            else:
                                if tracker.kill_question(q_num):
                                    tracker.log(f"Q{q_num} kill")
                        except ValueError:
                            tracker.log("Invalid number")
                    continue

            except Exception as e:
                tracker.log(f"KB Err: {e}")
                time.sleep(0.1)
                break

    except Exception as e:
        console.print(f"\nKB Setup Error: {e}", style="red")
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


# -------------------------------------------------------------------------
# Streaming Request
# -------------------------------------------------------------------------
async def streaming_request(
    client, prompt, input_text, tracker, question_num, stop_event
):
    encoding = get_tokenizer()
    full_messages = [{"role": "user", "content": prompt + input_text}]
    prompt_tokens = len(encoding.encode(prompt + input_text))
    total_tokens = prompt_tokens
    tracker.update_token(question_num, total_tokens)
    buffer_update_question(question_num, tokens=total_tokens, status="active")

    full_response_parts = []
    usage_data = None
    stream = None
    try:
        async with asyncio.timeout(3600):
            stream = await client.chat.completions.create(
                model=args.model_name,
                messages=full_messages,
                temperature=0.1,
                max_tokens=args.max_tokens,
                top_p=0.95,
                stream=True,
                #                extra_body={"repetition_penalty": 1.05},
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                if stop_event.is_set():
                    raise asyncio.CancelledError("Stop requested")
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_data = chunk.usage
                    continue
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content = delta.content or ""
                    thinking = (
                        getattr(delta, "reasoning_content", None)
                        or getattr(delta, "reasoning", None)
                        or ""
                    )
                    if content or thinking:
                        combined_text = thinking + content
                        delta_tokens = len(encoding.encode(combined_text))
                        total_tokens += delta_tokens
                        # Normal streaming update — rate calculation included
                        tracker.update_token(question_num, total_tokens)
                        buffer_update_question(
                            question_num, tokens=total_tokens, status="active"
                        )

                        if total_tokens > args.max_tokens:
                            raise MaxTokensExceeded(
                                f"Token count {total_tokens} exceeds max_tokens {args.max_tokens}"
                            )

                        if args.save_thinking:
                            full_response_parts.append(combined_text)
                        else:
                            if content:
                                full_response_parts.append(content)

            if usage_data is None:
                raise MissingUsageError("Server did not provide usage statistics")

            # FIX 3: Use count_only=True so the authoritative server token count is
            # stored without triggering a rate recalculation.  The client-side
            # estimated total_tokens may differ from usage_data.total_tokens, and
            # applying it as a normal delta update would cause a huge momentary spike
            # (large count jump / near-zero dt = infinite apparent tok/s).
            total_tokens = usage_data.total_tokens
            tracker.update_token(question_num, total_tokens, count_only=True)
            buffer_update_question(question_num, tokens=total_tokens, status="active")

        return "".join(full_response_parts), total_tokens

    except (
        asyncio.CancelledError,
        MaxTokensExceeded,
        MissingUsageError,
        asyncio.TimeoutError,
        Exception,
    ):
        raise
    finally:
        if stream is not None:
            try:
                await stream.close()
            except Exception:
                pass


# -------------------------------------------------------------------------
# Result Management
# -------------------------------------------------------------------------
def update_result(output_res_path):
    category_record = {}
    res = []
    success = False
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                for each in res:
                    cat = each["category"]
                    category_record.setdefault(cat, {"corr": 0.0, "wrong": 0.0})
                    if not each.get("pred"):
                        category_record[cat]["wrong"] += 1
                    elif each["pred"] == each["answer"]:
                        category_record[cat]["corr"] += 1
                    else:
                        category_record[cat]["wrong"] += 1
            success = True
        except Exception:
            time.sleep(2)
    return res, category_record


def merge_result(res, curr):
    for i, entry in enumerate(res):
        if (
            entry["question_id"] == curr["question_id"]
            and entry["question"] == curr["question"]
        ):
            res[i] = curr
            return res
    res.append(curr)
    return res


def save_res(res, output_res_path):
    uniq = []
    seen = set()
    for each in res:
        qid = each["question_id"]
        if qid not in seen:
            seen.add(qid)
            uniq.append(each)
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(uniq))


def save_summary(category_record, output_summary_path):
    total_corr = total_wrong = 0.0
    for cat, stats in category_record.items():
        if cat == "total":
            continue
        acc = (
            (stats["corr"] / (stats["corr"] + stats["wrong"]))
            if (stats["corr"] + stats["wrong"]) > 0
            else 0
        )
        category_record[cat]["acc"] = acc
        total_corr += stats["corr"]
        total_wrong += stats["wrong"]
    overall = {
        "corr": total_corr,
        "wrong": total_wrong,
        "acc": total_corr / (total_corr + total_wrong)
        if (total_corr + total_wrong) > 0
        else 0,
    }
    category_record["total"] = overall
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))


# -------------------------------------------------------------------------
# Evaluation Worker
# -------------------------------------------------------------------------
async def evaluate_subject_async(
    subject, test_data, dev_df, output_res_path, total_q, num_workers
):
    global RENDERING_ACTIVE
    buffer_clear()
    RENDERING_ACTIVE = True

    client = get_async_client()
    event_loop = asyncio.get_event_loop()
    tracker = ActiveQuestionTracker(
        subject, total_q, max_concurrent=num_workers, loop=event_loop
    )

    keyboard_thread = threading.Thread(
        target=keyboard_listener,
        args=(tracker, event_loop),
        daemon=True,
    )

    with Live(
        tracker.get_renderable(),
        console=console,
        screen=True,
        auto_refresh=False,
    ) as live:
        tracker.live = live

        keyboard_thread.start()
        await asyncio.sleep(0.5)

        existing_res, _ = update_result(output_res_path)
        processed_ids = {e["question_id"] for e in existing_res}

        rerun_maxtoken = getattr(args, "rerun_maxtoken", False)
        rerun_unknown = getattr(args, "rerun_unknown", False)
        maxtoken_ids = set()
        unknown_ids = set()
        if rerun_maxtoken:
            for ex in existing_res:
                if ex.get("model_outputs") == "MaxTokensExceeded":
                    maxtoken_ids.add(ex["question_id"])
            if maxtoken_ids:
                tracker.log(
                    f"Found {len(maxtoken_ids)} questions to rerun (MaxTokensExceeded)"
                )
            else:
                tracker.log("No MaxTokensExceeded questions found to rerun")
        if rerun_unknown:
            for ex in existing_res:
                if (
                    ex.get("pred") is None
                    and ex.get("model_outputs") != "MaxTokensExceeded"
                ):
                    unknown_ids.add(ex["question_id"])
            if unknown_ids:
                tracker.log(
                    f"Found {len(unknown_ids)} questions to rerun (Unknown prediction)"
                )
            else:
                tracker.log("No Unknown prediction questions found to rerun")

        pending = []
        rerun_ids = maxtoken_ids | unknown_ids
        for idx, each in enumerate(test_data):
            if each["question_id"] in processed_ids:
                if (rerun_maxtoken or rerun_unknown) and each[
                    "question_id"
                ] in rerun_ids:
                    pending.append((idx, each))
                    tracker.question_results[idx + 1] = "pending"
                else:
                    tracker.completed += 1
                    for ex in existing_res:
                        if ex["question_id"] == each["question_id"]:
                            pred = ex.get("pred")
                            ans = ex.get("answer")
                            res_status = (
                                "success"
                                if pred == ans and pred is not None
                                else "wrong"
                            )
                            tracker.question_results[idx + 1] = res_status
                            break
            else:
                pending.append((idx, each))

        last_start_time = 0
        start_lock = asyncio.Lock()
        running_tasks = set()
        task_to_qnum = {}
        stop_events = {}

        async def enforce_start_delay():
            nonlocal last_start_time
            async with start_lock:
                now = time.time()
                if last_start_time > 0:
                    delay = max(0.0, 2.0 - (now - last_start_time))
                    if delay > 0:
                        await asyncio.sleep(delay)
                last_start_time = time.time()

        async def process_one(question_num, each):
            nonlocal client, tracker
            stop_event = threading.Event()
            stop_events[question_num] = stop_event
            max_token_retries = args.retry
            max_wrong_retries = args.retry_wrong

            rerun_mode_maxtoken = getattr(args, "rerun_maxtoken", False)
            rerun_mode_unknown = getattr(args, "rerun_unknown", False)

            with file_lock:
                existing_res, _ = update_result(output_res_path)
                for ex in existing_res:
                    if each["question_id"] == ex["question_id"]:
                        should_rerun = False
                        if (
                            rerun_mode_maxtoken
                            and ex.get("model_outputs") == "MaxTokensExceeded"
                        ):
                            should_rerun = True
                        if (
                            rerun_mode_unknown
                            and ex.get("pred") is None
                            and ex.get("model_outputs") != "MaxTokensExceeded"
                        ):
                            should_rerun = True
                        if should_rerun:
                            pass
                        else:
                            tracker.complete_question(
                                question_num, token_count=0, success=False
                            )
                            return

            task = asyncio.current_task()
            tracker.start_question(question_num, task, stop_event)

            wrong_attempts = 0
            final_pred = None
            final_response = None
            final_token_count = 0

            while wrong_attempts <= max_wrong_retries and not GLOBAL_QUIT_REQUESTED:
                tracker.update_token(question_num, 0)
                tracker.set_stalled(question_num, False)
                token_exceed_attempts = 0
                request_success = False
                response = None
                token_count = None

                while True:
                    restart_flag, kill_flag = tracker.get_flags(question_num)

                    if kill_flag:
                        tracker.log(f"Q{question_num} killed")
                        with file_lock:
                            each["pred"] = None
                            each["model_outputs"] = "Killed by user"
                            res, _ = update_result(output_res_path)
                            res = merge_result(res, each)
                            save_res(res, output_res_path)
                        tracker.complete_question(
                            question_num, token_count=0, success=False
                        )
                        return

                    if restart_flag:
                        tracker.log(f"Q{question_num} restarting")
                        tracker.increment_retry_count(question_num)
                        token_exceed_attempts = 0
                        tracker.clear_flags(question_num)
                        stop_event.clear()
                        tracker.update_token(question_num, 0)
                        await asyncio.sleep(1)
                        continue

                    category = each["category"]
                    cot_examples = dev_df[category]
                    prompt = (
                        f"The following are multiple choice questions (with answers) about {category}. "
                        f"Think step by step and then output the answer in the format of "
                        f'"The answer is (X)" at the end.\n\n'
                    )
                    for ex in cot_examples:
                        prompt += format_example(
                            ex["question"], ex["options"], ex["cot_content"]
                        )
                    input_text = format_example(each["question"], each["options"])

                    try:
                        await enforce_start_delay()
                        response, token_count = await streaming_request(
                            client,
                            prompt,
                            input_text,
                            tracker,
                            question_num,
                            stop_event,
                        )
                        request_success = True
                        break

                    except asyncio.CancelledError:
                        if GLOBAL_QUIT_REQUESTED:
                            raise
                        restart_flag, kill_flag = tracker.get_flags(question_num)
                        if kill_flag:
                            tracker.log(f"Q{question_num} killed during req")
                            with file_lock:
                                each["pred"] = None
                                each["model_outputs"] = "Killed by user"
                                res, _ = update_result(output_res_path)
                                res = merge_result(res, each)
                                save_res(res, output_res_path)
                            tracker.complete_question(
                                question_num, token_count=0, success=False
                            )
                            return
                        if restart_flag:
                            continue
                        tracker.log(f"Q{question_num} cancelled")
                        tracker.increment_retry_count(question_num)
                        tracker.update_token(question_num, 0)
                        await asyncio.sleep(2)
                        continue

                    except MaxTokensExceeded:
                        token_exceed_attempts += 1
                        tracker.increment_retry_count(question_num)
                        if token_exceed_attempts >= max_token_retries:
                            tracker.log(f"Q{question_num} MaxTokens abort")
                            final_response = "MaxTokensExceeded"
                            request_success = False
                            break
                        else:
                            tracker.log(f"Q{question_num} MaxTokens retry")
                            tracker.update_token(question_num, 0)
                            await asyncio.sleep(min(2**token_exceed_attempts, 30))
                            continue

                    except (MissingUsageError, asyncio.TimeoutError, Exception) as e:
                        tracker.log(f"Q{question_num} Err: {str(e)[:30]}")
                        tracker.increment_retry_count(question_num)
                        tracker.update_token(question_num, 0)
                        await asyncio.sleep(5)
                        continue

                if not request_success:
                    final_pred = None
                    final_response = final_response or "Error"
                    break

                pred = extract_answer(response)
                if pred == each["answer"]:
                    final_pred = pred
                    final_response = response
                    final_token_count = token_count
                    break
                else:
                    wrong_attempts += 1
                    if wrong_attempts <= max_wrong_retries:
                        tracker.log(f"Q{question_num} wrong ans, retrying")
                        continue
                    else:
                        final_pred = pred
                        final_response = response
                        final_token_count = token_count
                        break

            if GLOBAL_QUIT_REQUESTED:
                return

            with file_lock:
                res, category_record = update_result(output_res_path)
                category_record.setdefault(subject, {"corr": 0.0, "wrong": 0.0})
                each["pred"] = final_pred
                each["model_outputs"] = final_response
                if final_pred is not None and final_pred == each["answer"]:
                    category_record[subject]["corr"] += 1
                else:
                    category_record[subject]["wrong"] += 1
                res = merge_result(res, each)
                save_res(res, output_res_path)
                save_summary(
                    category_record,
                    os.path.join(args.output_dir, f"{subject}_summary.json"),
                )

            if final_pred is not None and final_pred == each["answer"]:
                tracker.complete_question(
                    question_num, token_count=final_token_count, success=True
                )
            else:
                tracker.complete_question(question_num, token_count=0, success=False)

        # ---- Main scheduling loop ----
        while (pending or running_tasks) and not GLOBAL_QUIT_REQUESTED:
            if not INPUT_PROMPT_ACTIVE:
                tracker.render_screen()

            while (
                len(running_tasks) < num_workers
                and not tracker.any_stalled()
                and pending
                and not GLOBAL_QUIT_REQUESTED
            ):
                idx, each = pending.pop(0)
                q_num = idx + 1
                task = asyncio.create_task(process_one(q_num, each))
                task_to_qnum[task] = q_num
                running_tasks.add(task)

            if running_tasks:
                done, _ = await asyncio.wait(
                    running_tasks, timeout=0.2, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    running_tasks.discard(t)
                    q_num = task_to_qnum.pop(t, None)
                    if q_num:
                        stop_events.pop(q_num, None)
            else:
                await asyncio.sleep(0.1)

        if GLOBAL_QUIT_REQUESTED:
            tracker.stop_all_questions()
            for t in running_tasks:
                t.cancel()
            if running_tasks:
                await asyncio.gather(*running_tasks, return_exceptions=True)
        else:
            if running_tasks:
                await asyncio.wait(running_tasks)

        tracker.render_screen()
    # Live context exits here — screen restored


# -------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--model_name", "-m", type=str, default="local")
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    parser.add_argument("--url", "-u", type=str, default="http://127.0.0.1:8080/")
    parser.add_argument("--num_workers", "-n", type=int, default=4)
    parser.add_argument("--retry", "-r", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--retry_wrong", type=int, default=0)
    parser.add_argument("--save-thinking", action="store_true", default=False)
    parser.add_argument(
        "--rerun-maxtoken",
        action="store_true",
        default=False,
        help="Rerun questions that previously failed with MaxTokensExceeded",
    )
    parser.add_argument(
        "--rerun-unknown",
        action="store_true",
        default=False,
        help="Rerun questions where prediction was Unknown (no answer extracted)",
    )
    args = parser.parse_args()

    assigned_subjects = (
        [] if args.assigned_subjects == "all" else args.assigned_subjects.split(",")
    )
    os.makedirs(args.output_dir, exist_ok=True)

    test_df, dev_df = load_mmlu_pro()
    if not assigned_subjects:
        assigned_subjects = list(test_df.keys())

    console.print(f"Assigned subjects: {', '.join(assigned_subjects)}")
    if args.rerun_maxtoken:
        console.print(
            "[yellow]--rerun-maxtoken enabled: will retry questions that previously failed with MaxTokensExceeded[/yellow]"
        )
    if args.rerun_unknown:
        console.print(
            "[yellow]--rerun-unknown enabled: will retry questions where prediction was Unknown[/yellow]"
        )

    try:
        import uvloop

        uvloop.install()
        console.print("Using uvloop")
    except ImportError:
        pass

    for subject in assigned_subjects:
        if GLOBAL_QUIT_REQUESTED:
            console.print("\n[red]Evaluation terminated by user[/red]")
            break

        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, f"{subject}_result.json")
        total_q = len(test_data)
        SCROLL_OFFSET = 0
        AUTO_SCROLL = True

        try:
            asyncio.run(
                evaluate_subject_async(
                    subject,
                    test_data,
                    dev_df,
                    output_res_path,
                    total_q,
                    args.num_workers,
                )
            )
        except KeyboardInterrupt:
            console.print("\n[red]Interrupted by user[/red]")
            GLOBAL_QUIT_REQUESTED = True
            break
        except Exception as e:
            console.print(f"\nError processing {subject}: {e}", style="red")
            import traceback

            traceback.print_exc()
            continue

        summary_path = os.path.join(args.output_dir, f"{subject}_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                    if "total" in summary:
                        acc = summary["total"]["acc"] * 100
                        corr = int(summary["total"]["corr"])
                        wrong = int(summary["total"]["wrong"])
                        console.print(
                            f"\n  [bold]Final Accuracy for {subject}: "
                            f"{acc:.2f}% ({corr}/{corr + wrong})[/bold]"
                        )
            except Exception:
                pass
