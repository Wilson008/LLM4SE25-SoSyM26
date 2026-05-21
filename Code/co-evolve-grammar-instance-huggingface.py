import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI


DEFAULT_MODEL = "Qwen/Qwen3-Coder-Next:novita"
DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_CASES_ROOT = "Code/Step_3_Case_Languages"
DEFAULT_OUTPUT_TEMPLATE = "results_with_adjusted_prompt/instance_2_extra_qwen3_coder_next_{run}.txt"
DEFAULT_TIMING_FILE = "results_with_adjusted_prompt/qwen3_coder_next_time.csv"
CASE_LANGUAGE_DIRS = [
    "Code/Step_3_Case_Languages/CheckerDSL",
    "Code/Step_3_Case_Languages/eclipse-typescript-xtext",
    "Code/Step_3_Case_Languages/elite-se.xtext.languages.plantuml",
    "Code/Step_3_Case_Languages/isis-script",
    "Code/Step_3_Case_Languages/JSFLibraryGenerator",
    "Code/Step_3_Case_Languages/majordomo",
    "Code/Step_3_Case_Languages/mongoBeans",
    "Code/Step_3_Case_Languages/smart-dsl",
    "Code/Step_3_Case_Languages/xtext-dnn",
    "Code/Step_3_Case_Languages/xtext-orm",
]


def read_file(file_path):
    """Read UTF-8 text from a file."""
    return Path(file_path).read_text(encoding="utf-8")


def stream_chat_completion(client, model, messages, max_tokens, temperature, top_p, top_k):
    """Stream a chat completion and return the accumulated text."""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={"top_k": top_k},
        stream=True,
    )

    result = []
    char_count = 0
    next_progress = 500

    print("Streaming output (. = 500 chars):")
    for chunk in stream:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content
        if content is None:
            continue

        result.append(content)
        char_count += len(content)

        while char_count >= next_progress:
            print(".", end="", flush=True)
            next_progress += 500

    print()
    return "".join(result).strip()


def sanitize_ascii(text):
    """Normalize common model punctuation to ASCII and remove remaining non-ASCII symbols."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2192": "->",
        "\u2713": "",
        "\u2714": "",
        "\u274c": "",
        "\ufe0f": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return "".join(char for char in text if char in "\n\r\t" or 32 <= ord(char) <= 126)


def chat_acknowledgement(client, model, messages, max_tokens, temperature, top_p, top_k):
    """Request a short acknowledgement and append it to the conversation."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={"top_k": top_k},
    )
    content = response.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": content})


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a grammar/instance co-evolution experiment through Hugging Face Inference Providers."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("HF_MODEL", DEFAULT_MODEL),
        help=f"Hugging Face model id, optionally suffixed with a provider. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("HF_BASE_URL", DEFAULT_BASE_URL),
        help=f"Hugging Face OpenAI-compatible API base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--case-dir",
        default="Code/Step_3_Case_Languages/CheckerDSL",
        help="Directory containing grammar_1, grammar_2, and instance_1 files.",
    )
    parser.add_argument(
        "--grammar-1",
        default=None,
        help="Grammar 1 filename, relative to --case-dir unless an absolute path is provided.",
    )
    parser.add_argument(
        "--grammar-2",
        default=None,
        help="Grammar 2 filename, relative to --case-dir unless an absolute path is provided.",
    )
    parser.add_argument(
        "--instance-1",
        default=None,
        help="Instance 1 filename, relative to --case-dir unless an absolute path is provided.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_TEMPLATE,
        help="Output filename, relative to --case-dir unless an absolute path is provided.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs for the selected case.",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Run all case-language directories under --cases-root.",
    )
    parser.add_argument(
        "--cases-root",
        default=DEFAULT_CASES_ROOT,
        help=f"Root directory for --all-cases. Default: {DEFAULT_CASES_ROOT}",
    )
    parser.add_argument(
        "--timing-output",
        default=DEFAULT_TIMING_FILE,
        help="Timing CSV filename, relative to --case-dir unless an absolute path is provided.",
    )
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument(
        "--sanitize-ascii",
        action="store_true",
        help="Normalize generated text to ASCII before saving it.",
    )
    return parser


def resolve_path(case_dir, file_name):
    path = Path(file_name)
    if path.is_absolute():
        return path
    return Path(case_dir) / path


def find_single_file(case_dir, pattern):
    matches = sorted(Path(case_dir).glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {pattern} in {case_dir}, found {len(matches)}.")
    return matches[0].name


def resolve_case_files(args, case_dir):
    grammar_1 = args.grammar_1 or find_single_file(case_dir, "grammar_1*.txt")
    grammar_2 = args.grammar_2 or find_single_file(case_dir, "grammar_2*.txt")
    instance_1 = args.instance_1 or find_single_file(case_dir, "instance_1*.txt")
    return grammar_1, grammar_2, instance_1


def format_output_name(output_template, run_number):
    return output_template.format(run=run_number)


def append_timing(timing_file, row):
    timing_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not timing_file.exists()
    with timing_file.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_dir",
                "run",
                "model",
                "generation_start",
                "generation_end",
                "duration_seconds",
                "output_file",
                "output_lines",
                "output_characters",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def analyze_grammar_evolution(args):
    """Analyze grammar evolution and generate an evolved instance with a Hugging Face model."""
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        raise RuntimeError("HF_TOKEN is not set. Create a Hugging Face access token and set it before running.")

    start_time = datetime.now()
    print(f"Program started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")

    grammar_1_name, grammar_2_name, instance_1_name = resolve_case_files(args, args.case_dir)
    grammar_1_file = resolve_path(args.case_dir, grammar_1_name)
    grammar_2_file = resolve_path(args.case_dir, grammar_2_name)
    instance_1_file = resolve_path(args.case_dir, instance_1_name)
    output_file = resolve_path(args.case_dir, args.output)
    timing_file = resolve_path(args.case_dir, args.timing_output)

    grammar_1 = read_file(grammar_1_file)
    grammar_2 = read_file(grammar_2_file)
    instance_1 = read_file(instance_1_file)

    input_lines = len(instance_1.splitlines())
    print(f"Instance 1 has {input_lines} lines")
    print(f"Setting max_tokens={args.max_tokens} for final step")

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    messages = []

    print("\n=== Step 1: Sending Grammar 1 ===")
    messages.append(
        {
            "role": "user",
            "content": f"Here is the initial version of the grammar (Grammar 1). Please remember this for future reference:\n\n{grammar_1}",
        }
    )
    chat_acknowledgement(client, args.model, messages, 512, args.temperature, args.top_p, args.top_k)
    print("Grammar 1 acknowledged")

    print("\n=== Step 2: Sending Grammar 2 ===")
    messages.append(
        {
            "role": "user",
            "content": f"Here is the updated version of the grammar (Grammar 2). Please remember this and analyze the differences from Grammar 1:\n\n{grammar_2}",
        }
    )
    chat_acknowledgement(client, args.model, messages, 512, args.temperature, args.top_p, args.top_k)
    print("Grammar 2 acknowledged")

    print("\n=== Step 3: Sending Instance 1 ===")
    messages.append(
        {
            "role": "user",
            "content": f"Here is an instance of Grammar 1 (Instance 1). Please remember this for future reference:\n\n{instance_1}",
        }
    )
    chat_acknowledgement(client, args.model, messages, 1024, args.temperature, args.top_p, args.top_k)
    print("Instance 1 acknowledged")

    print("\n=== Step 4: Generating Instance 2 ===")
    final_prompt = """
grammar_1 is the initial grammar of the DSL. We evolved it to get grammar_2. instance_1 was originally a text instance that followed grammar_1. Now I want you to analyze the differences between the two versions of the grammar, and based on this difference, modify instance_1 and get instance_2, which will follow grammar_2. Please address the following things:
1. When evolving the instance, please do not omit any mandatory elements, such as characters enclosed by single quotes
2. If grammar_2 adds a new grammar rule or a new attribute that is optional or in an "OR" relationship (i.e., |), then please do not instantiate it.
3. Do not miss or add any auxiliary information in the instance, e.g., comments, formats (white space, indents, tabs, empty lines, etc.).
4. When evolving the instance, pay special attention to grammar rules that have been deleted in the evolved grammar. If a grammar rule no longer exists in Grammar 2, you must remove all instances of that rule from the evolved instance.
5. Always output your best attempt at the evolved instance, even if incomplete. Never return an empty result.
""".strip()

    messages.append({"role": "user", "content": final_prompt})

    generation_start = datetime.now()
    step_start = time.time()
    instance_2 = stream_chat_completion(
        client,
        args.model,
        messages,
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.top_k,
    )
    if args.sanitize_ascii:
        instance_2 = sanitize_ascii(instance_2)
    generation_end = datetime.now()
    step_duration = time.time() - step_start

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(instance_2, encoding="utf-8")

    output_lines = len(instance_2.splitlines())
    output_chars = len(instance_2)
    end_time = datetime.now()

    print(f"\n{'=' * 60}")
    print(f"Step 4 completed in {step_duration:.2f} seconds")
    print(f"Output lines: {output_lines}")
    print(f"Output characters: {output_chars}")
    print(f"{'=' * 60}")
    print(f"\nProgram ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {(end_time - start_time).total_seconds():.2f} seconds")
    print(f"\nInstance 2 has been saved to '{output_file}'.")

    append_timing(
        timing_file,
        {
            "case_dir": args.case_dir,
            "run": getattr(args, "run_number", ""),
            "model": args.model,
            "generation_start": generation_start.strftime("%Y-%m-%d %H:%M:%S"),
            "generation_end": generation_end.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": f"{step_duration:.2f}",
            "output_file": str(output_file),
            "output_lines": output_lines,
            "output_characters": output_chars,
        },
    )
    print(f"Timing has been appended to '{timing_file}'.")

    if output_lines < input_lines * 0.8:
        print("\nWARNING: Output may be truncated because it has significantly fewer lines than input.")
    else:
        print("\nOutput appears complete.")


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    if parsed_args.all_cases:
        output_template = parsed_args.output
        for case_dir in CASE_LANGUAGE_DIRS:
            for run_number in range(1, parsed_args.runs + 1):
                parsed_args.case_dir = str(case_dir)
                parsed_args.grammar_1 = None
                parsed_args.grammar_2 = None
                parsed_args.instance_1 = None
                parsed_args.output = format_output_name(output_template, run_number)
                parsed_args.run_number = run_number
                analyze_grammar_evolution(parsed_args)
    else:
        output_template = parsed_args.output
        for run_number in range(1, parsed_args.runs + 1):
            parsed_args.output = format_output_name(output_template, run_number)
            parsed_args.run_number = run_number
            analyze_grammar_evolution(parsed_args)
