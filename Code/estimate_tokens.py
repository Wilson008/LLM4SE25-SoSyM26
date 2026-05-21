"""
Token Consumption Estimator for LLM-based DSL Co-evolution
===========================================================
This script estimates the number of input and output tokens for each case language,
using the tiktoken library (OpenAI's tokenizer, also a reasonable approximation for Claude).

INSTRUCTIONS:
1. Install dependency: pip install tiktoken
2. Set BASE_DIR to the path of your Step_3_Case_Languages folder.
3. Run: python estimate_tokens.py

FILE NAMING ASSUMPTIONS (based on your repository structure):
  - Grammar 1:      grammar_1_*.txt
  - Grammar 2:      grammar_2_*.txt
  - Instance 1:     instance_1_*.txt
  - Claude outputs: instance_2_gen_claude_*.txt  (10 files)
  - GPT outputs:    instance_2_gen_openai_*.txt  (10 files)
"""

import os
import glob
import tiktoken

# ============================================================
# CONFIGURATION: Set your base directory here
# ============================================================

BASE_DIR = "C:/01.Work/09.GitHub_Repos/LLM4SE25-SoSyM26/Code/Step_3_Case_Languages"

# Case language folder names (as they appear on disk)
CASE_LANGUAGE_FOLDERS = [
    "xtext-orm",
    "xtext-dnn",
    "smart-dsl",
    "mongoBeans",
    "elite-se.xtext.languages.plantuml",
    "isis-script",
    "CheckerDSL",
    "eclipse-typescript-xtext",
    "JSFLibraryGenerator",
    "majordomo",
]

# Display names for the table (matches your paper)
DISPLAY_NAMES = {
    "xtext-orm":                            "xtext-orm",
    "xtext-dnn":                            "xtext-dnn",
    "smart-dsl":                            "smart-dsl",
    "mongoBeans":                           "mongoBeans",
    "elite-se.xtext.languages.plantuml":    "plantuml",
    "isis-script":                          "isis-script",
    "CheckerDSL":                           "CheckerDSL",
    "eclipse-typescript-xtext":             "eclipse-typescript-xtext",
    "JSFLibraryGenerator":                  "JSFLibraryGenerator",
    "majordomo":                            "majordomo",
}

# The final prompt text
PROMPT_TEXT = """grammar_1 is the initial grammar of the DSL. We evolved it to get grammar_2. instance_1 was
originally a text instance that followed grammar_1. Now I want you to analyze the differences
between the two versions of the grammar and, based on this difference, modify instance_1 and get
instance_2, which will follow grammar_2. Please address the following things:
1. When evolving the instance, please do not omit any mandatory elements, such as characters
enclosed by single quotes.
2. If grammar_2 adds a new grammar rule or a new attribute that is optional or in an "OR"
relationship (i.e., |), then please do not instantiate it.
3. Do not miss or add any auxiliary information in the instance, e.g., comments, formats (white
space, indents, tabs, empty lines, etc.)."""

# Hint messages sent before each file (from your Python script workflow)
HINT_GRAMMAR_1  = "Here is the initial version of the grammar (i.e., Grammar 1). Please remember this for future reference."
HINT_GRAMMAR_2  = "Here is the evolved version of the grammar (i.e., Grammar 2). Please remember this for future reference."
HINT_INSTANCE_1 = "Here is the instance (i.e., Instance 1) that conforms to Grammar 1. Please remember this for future reference."

# ============================================================
# TOKENIZER SETUP
# ============================================================
# Using cl100k_base (GPT-4 encoding; reasonable approximation for both GPT and Claude)
enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def read_first_match(folder: str, pattern: str) -> str:
    """Read the first file matching a glob pattern in a folder."""
    matches = glob.glob(os.path.join(folder, pattern))
    if not matches:
        print(f"  [WARNING] No file matching '{pattern}' in: {folder}")
        return ""
    if len(matches) > 1:
        print(f"  [INFO] Multiple files matching '{pattern}' in {folder}, using: {os.path.basename(matches[0])}")
    with open(matches[0], "r", encoding="utf-8") as f:
        return f.read()


def avg_tokens_from_pattern(folder: str, pattern: str) -> float:
    """Compute average token count across all files matching a pattern."""
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        print(f"  [WARNING] No files matching '{pattern}' in: {folder}")
        return 0.0
    token_counts = [count_tokens(open(fp, "r", encoding="utf-8").read()) for fp in files]
    return sum(token_counts) / len(token_counts)


# ============================================================
# MAIN ESTIMATION
# ============================================================

prompt_tokens = count_tokens(PROMPT_TEXT)
hint_tokens   = (count_tokens(HINT_GRAMMAR_1)
                 + count_tokens(HINT_GRAMMAR_2)
                 + count_tokens(HINT_INSTANCE_1))

print("=" * 80)
print(f"Fixed tokens per run:")
print(f"  Prompt text:    {prompt_tokens} tokens")
print(f"  Hint messages:  {hint_tokens} tokens")
print(f"  Combined fixed: {prompt_tokens + hint_tokens} tokens")
print("=" * 80)
print()

col1, col2, col3, col4 = 35, 12, 12, 12
header = (f"{'Case Language':<{col1}} {'Input':>{col2}} "
          f"{'Output(Claude)':>{col3}} {'Output(GPT)':>{col4}}")
print(header)
print("-" * len(header))

total_input   = 0
total_claude  = 0
total_gpt     = 0
n = 0

for folder_name in CASE_LANGUAGE_FOLDERS:
    folder_path  = os.path.join(BASE_DIR, folder_name)
    display_name = DISPLAY_NAMES[folder_name]

    g1 = read_first_match(folder_path, "grammar_1_*.txt")
    g2 = read_first_match(folder_path, "grammar_2_*.txt")
    i1 = read_first_match(folder_path, "instance_1_*.txt")

    input_tokens = (hint_tokens
                    + count_tokens(g1)
                    + count_tokens(g2)
                    + count_tokens(i1)
                    + prompt_tokens)

    output_claude = avg_tokens_from_pattern(folder_path, "instance_2_gen_claude_*.txt")
    output_gpt    = avg_tokens_from_pattern(folder_path, "instance_2_gen_openai_*.txt")

    print(f"{display_name:<{col1}} {input_tokens:>{col2}} "
          f"{output_claude:>{col3}.0f} {output_gpt:>{col4}.0f}")

    total_input  += input_tokens
    total_claude += output_claude
    total_gpt    += output_gpt
    n += 1

print("-" * len(header))
print(f"{'Average':<{col1}} {total_input/n:>{col2}.0f} "
      f"{total_claude/n:>{col3}.0f} {total_gpt/n:>{col4}.0f}")
print()
print("Notes:")
print("  - Input tokens are identical for both LLMs (same prompt and input files).")
print("  - Output tokens are averaged over 10 runs per case language.")
print("  - Token counts are estimated using tiktoken (cl100k_base encoding).")
print("  - This is an approximation; actual billed tokens may differ slightly.")