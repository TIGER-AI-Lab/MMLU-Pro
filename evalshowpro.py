#!/usr/bin/env python3
import json
import os
import glob
import sys
import curses
from typing import List, Dict, Any

class EnhancedQuestionBrowser:
    def __init__(self, results_dir: str = "eval_results/"):
        self.results_dir = results_dir
        self.all_questions = {}
        self.wrong_questions = []
        self.current_index = 0
        self.model_output_expanded = False
        self.model_output_page = 0
        self.model_output_total_pages = 0
        self.categories = []
        self.current_category = "all"
        self.stdscr = None
        self.needs_redraw = True

    def load_questions(self):
        """Load all questions from result files organized by category"""
        result_files = glob.glob(os.path.join(self.results_dir, "*_result.json"))

        if not result_files:
            print(f"No result files found in {self.results_dir}")
            return False

        self.all_questions = {}

        for file_path in result_files:
            try:
                category = os.path.basename(file_path).replace("_result.json", "")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Clean the data by ensuring all string fields have proper values
                        cleaned_data = [self._clean_question(q) for q in data]
                        self.all_questions[category] = cleaned_data
                        print(f"Loaded {len(cleaned_data)} questions from {category}")
                    else:
                        print(f"Unexpected format in {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        self.categories = list(self.all_questions.keys())
        self.update_wrong_questions()

        print(f"\nLoaded {len(self.wrong_questions)} wrong questions from {len(self.categories)} categories")
        return True

    def _clean_question(self, question: Dict) -> Dict:
        """Ensure all string fields in a question have proper values"""
        # Define default values for each field
        defaults = {
            'category': 'Unknown',
            'question_id': 'Unknown',
            'question': 'No question text',
            'options': [],
            'answer': 'Unknown',
            'pred': 'Unknown',
            'answer_index': 0,
            'cot_content': '',
            'model_outputs': '',
            'src': 'Unknown'
        }

        cleaned = question.copy()
        for field, default in defaults.items():
            if field in cleaned:
                if cleaned[field] is None:
                    cleaned[field] = default
                elif field == 'options' and not isinstance(cleaned[field], list):
                    cleaned[field] = []
            else:
                cleaned[field] = default

        return cleaned

    def update_wrong_questions(self):
        """Update the wrong questions list based on current category filter"""
        if self.current_category == "all":
            # Combine all questions from all categories
            all_questions = []
            for category_questions in self.all_questions.values():
                all_questions.extend(category_questions)
        else:
            all_questions = self.all_questions.get(self.current_category, [])

        self.wrong_questions = [
            q for q in all_questions
            if q.get('answer') != q.get('pred')
        ]
        # Ensure current_index is within bounds
        self._ensure_valid_index()

    def _ensure_valid_index(self):
        """Ensure current_index is always valid"""
        if not self.wrong_questions:
            self.current_index = 0
        else:
            # Clamp current_index to valid range
            self.current_index = max(0, min(self.current_index, len(self.wrong_questions) - 1))

    def get_category_stats(self):
        """Get statistics for each category"""
        stats = {}
        for category, questions in self.all_questions.items():
            wrong_count = len([q for q in questions if q.get('answer') != q.get('pred')])
            total_count = len(questions)
            stats[category] = {
                'total': total_count,
                'wrong': wrong_count,
                'accuracy': ((total_count - wrong_count) / total_count * 100) if total_count > 0 else 0
            }
        return stats

    def safe_get_text(self, text, default="N/A"):
        """Safely get text, handling None values"""
        if text is None:
            return default
        return str(text)

    def truncate_text(self, text: str, max_length: int = 80) -> str:
        """Truncate long text for display"""
        safe_text = self.safe_get_text(text, "")
        if len(safe_text) > max_length:
            return safe_text[:max_length] + "..."
        return safe_text

    def format_model_output(self, text: str, width: int) -> List[str]:
        """Format model output with better formatting for large content"""
        safe_text = self.safe_get_text(text, "")
        if not safe_text:
            return [""]

        # Split into paragraphs first
        paragraphs = safe_text.split('\n')
        wrapped_lines = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                wrapped_lines.append("")
                continue

            words = paragraph.split()
            current_line = []
            current_length = 0

            for word in words:
                word_length = len(word)
                # If adding this word would exceed width, start a new line
                if current_length + word_length + (1 if current_line else 0) > width - 4:  # -4 for indentation
                    if current_line:
                        wrapped_lines.append("  " + " ".join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    current_line.append(word)
                    current_length += word_length + (1 if current_line else 0)

            if current_line:
                wrapped_lines.append("  " + " ".join(current_line))

        return wrapped_lines

    def calculate_model_output_pages(self, formatted_lines: List[str], start_row: int) -> int:
        """Calculate total pages for model output based on available screen space"""
        if not formatted_lines:
            return 1

        # Calculate available rows for model output (leave space for page info and commands)
        available_rows = curses.LINES - start_row - 6  # More conservative estimate
        if available_rows <= 0:
            available_rows = 1

        lines_per_page = available_rows
        total_pages = (len(formatted_lines) + lines_per_page - 1) // lines_per_page

        return max(1, total_pages)

    def safe_addstr(self, row: int, col: int, text: str, attr=0):
        """Safely add string to screen, handling potential errors"""
        try:
            # Ensure we don't go beyond screen bounds
            if row < 0 or row >= curses.LINES:
                return row
            if col < 0 or col >= curses.COLS:
                return row

            # Truncate text if it would go beyond screen width
            max_len = curses.COLS - col
            if max_len <= 0:
                return row

            safe_text = self.safe_get_text(text, "")
            display_text = safe_text[:max_len]
            self.stdscr.addstr(row, col, display_text, attr)
            return row + 1
        except curses.error:
            # If we get a curses error, just move to next line
            return row + 1

    def display_question(self):
        """Display current question with expandable fields using curses"""
        try:
            if not self.wrong_questions:
                self.stdscr.clear()
                self.stdscr.addstr(0, 0, "No wrong questions found in current category!")
                self.stdscr.refresh()
                return

            # Ensure current_index is within bounds
            self._ensure_valid_index()

            q = self.wrong_questions[self.current_index]
            self.stdscr.clear()

            # Header with boundary indicators
            stats = self.get_category_stats()

            # Add boundary indicators
            boundary_indicator = ""
            if self.current_index == 0 and len(self.wrong_questions) > 1:
                boundary_indicator = " [FIRST]"
            elif self.current_index == len(self.wrong_questions) - 1 and len(self.wrong_questions) > 1:
                boundary_indicator = " [LAST]"
            elif len(self.wrong_questions) == 1:
                boundary_indicator = " [ONLY ONE]"

            header = f"MMLU-Pro Wrong Answers Browser - {self.current_category.upper()} - Question {self.current_index + 1}/{len(self.wrong_questions)}{boundary_indicator}"
            self.safe_addstr(0, 0, header, curses.A_BOLD)

            # Category summary
            category_line = "Categories: "
            for i, category in enumerate(self.categories):
                stat = stats[category]
                if category == self.current_category:
                    category_line += f"[{category}: {stat['wrong']}/{stat['total']}] "
                else:
                    category_line += f"{category}: {stat['wrong']}/{stat['total']} "
            if self.current_category == "all":
                total_wrong = sum(stat['wrong'] for stat in stats.values())
                total_questions = sum(stat['total'] for stat in stats.values())
                category_line += f" [ALL: {total_wrong}/{total_questions}]"

            self.safe_addstr(1, 0, self.truncate_text(category_line, curses.COLS-1))

            # Separator
            self.safe_addstr(2, 0, "=" * min(80, curses.COLS-1), curses.A_BOLD)

            # Current question info
            row = 3
            row = self.safe_addstr(row, 0, f"Category: {self.safe_get_text(q.get('category'))}", curses.A_BOLD)
            row = self.safe_addstr(row, 0, f"Question ID: {self.safe_get_text(q.get('question_id'))}")
            row = self.safe_addstr(row, 0, f"Source: {self.safe_get_text(q.get('src'))}")

            # Correct Answer with safe handling
            row = self.safe_addstr(row, 0, "Correct Answer: ")
            answer_text = self.safe_get_text(q.get('answer'))
            try:
                self.stdscr.addstr(answer_text, curses.color_pair(2))  # Green
            except:
                self.stdscr.addstr("N/A", curses.color_pair(2))
            row += 1

            # Model Prediction with safe handling
            row = self.safe_addstr(row, 0, "Model Prediction: ")
            pred_text = self.safe_get_text(q.get('pred'))
            try:
                self.stdscr.addstr(pred_text, curses.color_pair(1))  # Red
            except:
                self.stdscr.addstr("N/A", curses.color_pair(1))
            row += 1

            row = self.safe_addstr(row, 0, f"Answer Index: {self.safe_get_text(q.get('answer_index'))}")
            row += 2

            # Question text
            question_text = q.get('question', '')
            row = self.safe_addstr(row, 0, f"Question: {self.truncate_text(question_text, curses.COLS-20)}", curses.A_BOLD | curses.color_pair(3))
            row += 1

            # Options (in one line)
            options = q.get('options', [])
            if options and isinstance(options, list):
                # Filter out None values from options
                safe_options = [self.safe_get_text(opt, "Empty option") for opt in options]
                options_text = " | ".join([f"{i+1}. {opt}" for i, opt in enumerate(safe_options)])
                row = self.safe_addstr(row, 0, f"Options: {self.truncate_text(options_text, curses.COLS-10)}", curses.A_BOLD)
                row += 2
            elif options:
                # Handle case where options is not a list
                row = self.safe_addstr(row, 0, f"Options: Invalid format", curses.A_BOLD)
                row += 2

            # Chain of Thought
            cot = q.get('cot_content', '')
            if cot:
                row = self.safe_addstr(row, 0, f"Chain of Thought: {self.truncate_text(cot, curses.COLS-25)}", curses.A_BOLD | curses.color_pair(4))
                row += 2

            # Store the starting row for model output
            model_output_start_row = row

            # Model Outputs - with toggle and paging
            model_outputs = q.get('model_outputs', '')
            if model_outputs:
                status = "EXPANDED" if self.model_output_expanded else "COLLAPSED"
                row = self.safe_addstr(row, 0, f"Model Outputs [{status}]:", curses.A_BOLD | curses.color_pair(1))
                row += 1

                if self.model_output_expanded:
                    try:
                        formatted_lines = self.format_model_output(model_outputs, curses.COLS)

                        # Calculate total pages based on current screen layout
                        self.model_output_total_pages = self.calculate_model_output_pages(formatted_lines, model_output_start_row + 2)

                        # Ensure current page is within bounds
                        if self.model_output_page >= self.model_output_total_pages:
                            self.model_output_page = self.model_output_total_pages - 1
                        if self.model_output_page < 0:
                            self.model_output_page = 0

                        # Calculate lines per page
                        available_rows = curses.LINES - model_output_start_row - 8  # Conservative estimate
                        if available_rows <= 0:
                            available_rows = 1
                        lines_per_page = available_rows

                        # Display current page
                        start_idx = self.model_output_page * lines_per_page
                        end_idx = min(start_idx + lines_per_page, len(formatted_lines))

                        for i in range(start_idx, end_idx):
                            if row >= curses.LINES - 4:  # Leave room for page info and commands
                                row = self.safe_addstr(row, 0, "... (output truncated)")
                                break
                            row = self.safe_addstr(row, 0, formatted_lines[i])

                        # Display page info
                        if self.model_output_total_pages > 1:
                            page_info = f"--- Page {self.model_output_page + 1}/{self.model_output_total_pages} ---"
                            row = self.safe_addstr(row, 0, page_info, curses.A_BOLD)
                            row += 1

                    except Exception as e:
                        # If model output is too large and causes issues, collapse it
                        self.model_output_expanded = False
                        self.model_output_page = 0
                        row = self.safe_addstr(row, 0, "Output too large - auto collapsed")
                else:
                    row = self.safe_addstr(row, 0, f"  {self.truncate_text(model_outputs, curses.COLS-20)}")
                    row += 1

            # Commands
            row = curses.LINES - 3
            self.safe_addstr(row, 0, "Commands:", curses.A_BOLD)
            row += 1

            if self.model_output_expanded:
                cmd_text = "←/→:Navigate  ↑/↓:Page Output  PgUp/PgDn:Jump  Enter:Category  S:Search  Q:Quit"
            else:
                cmd_text = "←/→:Navigate  ↑/↓:Toggle Output  PgUp/PgDn:Jump  Enter:Category  S:Search  Q:Quit"

            self.safe_addstr(row, 0, cmd_text)

            self.stdscr.refresh()
            self.needs_redraw = False  # We just redrew

        except Exception as e:
            # If we get any error in display, show the error but don't reset to first question
            self.stdscr.clear()
            self.stdscr.addstr(0, 0, f"Display error at question {self.current_index + 1}: {str(e)}")
            self.stdscr.addstr(2, 0, "Press any key to continue...")
            self.stdscr.refresh()
            self.stdscr.getch()
            self.needs_redraw = True

    def wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to fit terminal width"""
        safe_text = self.safe_get_text(text, "")
        if not safe_text:
            return [""]
        words = safe_text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def show_category_selection(self):
        """Show category selection with arrow key navigation"""
        options = self.categories + ["ALL CATEGORIES"]
        current_selection = len(options) - 1 if self.current_category == "all" else self.categories.index(self.current_category)

        while True:
            self.stdscr.clear()
            self.stdscr.addstr(0, 0, "SELECT CATEGORY (Use ↑/↓ arrows, Enter to select)", curses.A_BOLD | curses.A_UNDERLINE)

            stats = self.get_category_stats()

            for i, option in enumerate(options):
                if i == current_selection:
                    self.stdscr.addstr(2 + i, 2, f"> {option}", curses.A_REVERSE)
                else:
                    self.stdscr.addstr(2 + i, 2, f"  {option}")

                # Add stats for categories
                if i < len(self.categories):
                    stat = stats[self.categories[i]]
                    self.stdscr.addstr(2 + i, 30, f"{stat['wrong']}/{stat['total']} wrong ({stat['accuracy']:.1f}% accuracy)")
                elif option == "ALL CATEGORIES":
                    total_wrong = sum(stat['wrong'] for stat in stats.values())
                    total_questions = sum(stat['total'] for stat in stats.values())
                    total_accuracy = ((total_questions - total_wrong) / total_questions * 100) if total_questions > 0 else 0
                    self.stdscr.addstr(2 + i, 30, f"{total_wrong}/{total_questions} wrong ({total_accuracy:.1f}% accuracy)")

            self.stdscr.refresh()

            key = self.stdscr.getch()

            if key == curses.KEY_UP:
                current_selection = (current_selection - 1) % len(options)
            elif key == curses.KEY_DOWN:
                current_selection = (current_selection + 1) % len(options)
            elif key == ord('\n') or key == ord('\r'):  # Enter
                break
            elif key == ord('q') or key == 27:  # q or ESC to cancel
                return

        # Set the selected category
        if current_selection == len(options) - 1:
            self.current_category = "all"
        else:
            self.current_category = self.categories[current_selection]

        self.update_wrong_questions()
        self.needs_redraw = True

    def search_by_id(self):
        """Search for question by ID"""
        if not self.wrong_questions:
            return

        self.stdscr.clear()
        prompt = "SEARCH - Enter question ID: "
        self.stdscr.addstr(0, 0, prompt)
        self.stdscr.refresh()

        curses.echo()
        try:
            search_id = self.stdscr.getstr(0, len(prompt), 20).decode('utf-8').strip()
        finally:
            curses.noecho()

        if search_id:
            # Search for the question ID in wrong_questions
            found = False
            for i, q in enumerate(self.wrong_questions):
                question_id = self.safe_get_text(q.get('question_id'))
                if question_id.lower() == search_id.lower():
                    self.current_index = i
                    self.model_output_page = 0  # Reset to first page
                    found = True
                    break

            if not found:
                # Show not found message
                self.stdscr.clear()
                self.stdscr.addstr(0, 0, f"Question ID '{search_id}' not found!")
                self.stdscr.addstr(2, 0, "Press any key to continue...")
                self.stdscr.refresh()
                self.stdscr.getch()

        self.needs_redraw = True

    def show_help(self):
        """Show help screen"""
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, "HELP - MMLU-Pro Browser", curses.A_BOLD | curses.A_UNDERLINE)

        help_text = [
            "",
            "NAVIGATION:",
            "  ← / →    - Previous/Next question",
            "  PgUp/PgDn - Move 10 questions",
            "  Enter    - Select category",
            "  S        - Search by question ID",
            "",
            "VIEW CONTROLS:",
            "  ↑ / ↓    - Toggle model output (collapsed) or page through output (expanded)",
            "  Q        - Quit",
            "",
            "BOUNDARY INDICATORS:",
            "  [FIRST]  - You are at the first question",
            "  [LAST]   - You are at the last question",
            "  [ONLY ONE] - Only one question in current category",
            "",
            "Press any key to continue..."
        ]

        for i, line in enumerate(help_text):
            self.safe_addstr(2 + i, 0, line)

        self.stdscr.refresh()
        self.stdscr.getch()
        self.needs_redraw = True

    def handle_keypress(self):
        """Handle keypress with curses - with robust bounds checking"""
        try:
            key = self.stdscr.getch()

            # Handle navigation keys with strict bounds checking
            if key == curses.KEY_LEFT or key == ord('h'):
                if self.current_index > 0:
                    self.current_index -= 1
                    if self.model_output_expanded:
                        self.model_output_page = 0
                    self.needs_redraw = True

            elif key == curses.KEY_RIGHT or key == ord('l'):
                if self.current_index < len(self.wrong_questions) - 1:
                    self.current_index += 1
                    if self.model_output_expanded:
                        self.model_output_page = 0
                    self.needs_redraw = True

            elif key == curses.KEY_UP:
                if self.model_output_expanded:
                    # Page up in model output
                    if self.model_output_page > 0:
                        self.model_output_page -= 1
                        self.needs_redraw = True
                    else:
                        # On first page, up collapses
                        self.model_output_expanded = False
                        self.needs_redraw = True
                else:
                    # Toggle model output expansion
                    self.model_output_expanded = True
                    self.model_output_page = 0
                    self.needs_redraw = True

            elif key == curses.KEY_DOWN:
                if self.model_output_expanded:
                    # Page down in model output
                    if self.model_output_page < self.model_output_total_pages - 1:
                        self.model_output_page += 1
                        self.needs_redraw = True
                else:
                    # Toggle model output expansion
                    self.model_output_expanded = True
                    self.model_output_page = 0
                    self.needs_redraw = True

            elif key == curses.KEY_PPAGE:  # Page Up
                new_index = max(0, self.current_index - 10)
                if new_index != self.current_index:
                    self.current_index = new_index
                    if self.model_output_expanded:
                        self.model_output_page = 0
                    self.needs_redraw = True

            elif key == curses.KEY_NPAGE:  # Page Down
                new_index = min(len(self.wrong_questions) - 1, self.current_index + 10)
                if new_index != self.current_index:
                    self.current_index = new_index
                    if self.model_output_expanded:
                        self.model_output_page = 0
                    self.needs_redraw = True

            elif key == ord('\n') or key == ord('\r'):  # Enter
                self.show_category_selection()
                # needs_redraw is set in show_category_selection

            elif key == ord('s') or key == ord('S'):
                self.search_by_id()
                # needs_redraw is set in search_by_id

            elif key == ord('q') or key == ord('Q'):
                return False

            elif key == ord('h') or key == ord('H'):
                self.show_help()
                # needs_redraw is set in show_help

            # Unknown keys are ignored - no redraw needed

            return True

        except KeyboardInterrupt:
            return False
        except Exception as e:
            # On any error in key handling, mark for redraw and continue
            self.needs_redraw = True
            return True

    def init_curses(self):
        """Initialize curses"""
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        curses.curs_set(0)  # Hide cursor

        # Initialize colors
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_RED, -1)      # Wrong predictions
            curses.init_pair(2, curses.COLOR_GREEN, -1)    # Correct answers
            curses.init_pair(3, curses.COLOR_YELLOW, -1)   # Questions
            curses.init_pair(4, curses.COLOR_CYAN, -1)     # Chain of Thought

    def cleanup_curses(self):
        """Clean up curses"""
        if self.stdscr:
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.endwin()

    def run(self):
        """Main program loop with comprehensive error handling"""
        if not self.load_questions():
            return

        if not self.wrong_questions:
            print("No wrong answers found!")
            return

        try:
            self.init_curses()
            running = True

            # Initial display
            self.display_question()

            while running:
                try:
                    # Only redraw if needed
                    if self.needs_redraw:
                        self.display_question()

                    # Handle keypress (blocking)
                    running = self.handle_keypress()

                except Exception as e:
                    # Handle any display or keypress errors gracefully
                    self.stdscr.clear()
                    self.stdscr.addstr(0, 0, f"Unexpected error: {str(e)}")
                    self.stdscr.addstr(2, 0, "Attempting to recover...")
                    self.stdscr.refresh()
                    curses.napms(2000)

                    # Reset to safe state but don't jump to first question
                    self._ensure_valid_index()
                    self.needs_redraw = True
                    continue

        except KeyboardInterrupt:
            print("\nReceived interrupt signal, shutting down...")
        except Exception as e:
            print(f"Fatal error: {e}")
        finally:
            self.cleanup_curses()

def main():
    browser = EnhancedQuestionBrowser()
    browser.run()

if __name__ == "__main__":
    main()