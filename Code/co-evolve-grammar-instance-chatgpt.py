from openai import OpenAI
import time
from datetime import datetime

def read_file(file_path):
    """Read the content of a file"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def analyze_grammar_evolution_gpt():
    """Main function for GPT-5.2 based co-evolution"""
    
    # Record start time
    start_time = datetime.now()
    print(f"Program started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Set OpenAI API key
    client = OpenAI(api_key="")

    # File paths
    grammar_a_file = "Code\\Step_3_Case_Languages\\CheckerDSL\\grammar_1_20150503_55911bf.txt"
    grammar_b_file = "Code\\Step_3_Case_Languages\\CheckerDSL\\grammar_2_20150727_3fa6e6d.txt"
    instance_a_file = "Code\\Step_3_Case_Languages\\CheckerDSL\\instance_1_20250503_55911bf.txt"

    # Load grammar 1, grammar 2, and instance 1
    grammar_1 = read_file(grammar_a_file)
    grammar_2 = read_file(grammar_b_file)
    instance_1 = read_file(instance_a_file)

    # Check instance size and set appropriate max_completion_tokens
    instance_lines = len(instance_1.split('\n'))
    print(f"Instance 1 has {instance_lines} lines")
    
    # Set max_completion_tokens to a large value for GPT-5.2
    # GPT-5.2's exact limit needs verification, trying 100000 first
    final_max_tokens = 100000
    print(f"Setting max_completion_tokens={final_max_tokens} for final step")

    # Initialize the messages list for chain-of-thought prompting
    messages = []

    # Step 1: Send Grammar 1
    print("\n=== Step 1: Sending Grammar 1 ===")
    messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.append({
        "role": "user",
        "content": f"Here is the initial version of the grammar (Grammar 1). Please remember this for future reference:\n\n{grammar_1}"
    })

    response = client.chat.completions.create(
        model="gpt-5.2", 
        messages=messages, 
        max_completion_tokens=200
    )
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    print("Grammar 1 acknowledged")

    # Step 2: Send Grammar 2
    print("\n=== Step 2: Sending Grammar 2 ===")
    messages.append({
        "role": "user",
        "content": f"Here is the updated version of the grammar (Grammar 2). Please remember this and analyze the differences from Grammar 1:\n\n{grammar_2}"
    })

    response = client.chat.completions.create(
        model="gpt-5.2", 
        messages=messages, 
        max_completion_tokens=200
    )
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    print("Grammar 2 acknowledged")

    # Step 3: Send Instance 1
    print("\n=== Step 3: Sending Instance 1 ===")
    messages.append({
        "role": "user",
        "content": f"Here is an instance of Grammar 1 (Instance 1). Please remember this for future reference:\n\n{instance_1}"
    })

    response = client.chat.completions.create(
        model="gpt-5.2", 
        messages=messages, 
        max_completion_tokens=1000
    )
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    print("Instance 1 acknowledged")

    # Step 4: Generate Instance 2 with streaming
    print("\n=== Step 4: Generating Instance 2 (this may take 10-15 minutes) ===")
    final_prompt = """
        grammar_1 is the initial grammar of the DSL, we evolved it to get grammar_2, and instance_1 was originally a text instance that followed grammar_1. 
        Now I want you to analyze the differences between the two versions of the grammar, and based on this difference, modify instance_1 and get instance_2, which will follow grammar_2. 
        And please note the following things:
        1. When evolve the instance, please don't forget the symbol which is enclosed by single quotes.
        2. If grammar_2 add a new grammar rule or a new attribute which is optional or in an "OR" relationship (i.e., |), then please don't instantiate it.
        3. Don't miss or add any formats in the instance, e.g., comments, formats (white-space, indents, tabs, empty lines, etc.). 
        4. When evolving the instance, pay special attention to grammar rules that have been deleted in the evolved grammar. If a grammar rule no longer exists in Grammar 2, you must remove all instances of that rule from the evolved instance.
        5. Always output your best attempt at the evolved instance, even if incomplete. Never return an empty result.
    """
    messages.append({"role": "user", "content": final_prompt})

    # Use streaming to handle large output
    step4_start = time.time()
    
    try:
        stream = client.chat.completions.create(
            model="gpt-5.2",
            messages=messages,
            max_completion_tokens=final_max_tokens,
            stream=True
        )
        
        instance_2 = ""
        token_count = 0
        print("Streaming output (. = 500 chars):")
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                instance_2 += content
                token_count += len(content)
                
                # Progress indicator every 500 characters
                if token_count % 500 == 0:
                    print(".", end="", flush=True)
        
        print("\n")  # New line after progress dots
        
    except Exception as e:
        print(f"\nError during streaming: {e}")
        print("Trying non-streaming mode with reduced max_tokens...")
        
        # Fallback to non-streaming
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=messages,
            max_completion_tokens=16000  # Reduced fallback
        )
        instance_2 = response.choices[0].message.content.strip()
    
    step4_end = time.time()
    step4_duration = step4_end - step4_start
    
    # Statistics
    output_lines = len(instance_2.split('\n'))
    output_chars = len(instance_2)
    
    print(f"\n{'='*60}")
    print(f"Step 4 completed in {step4_duration:.2f} seconds")
    print(f"Output lines: {output_lines}")
    print(f"Output characters: {output_chars}")
    print(f"{'='*60}")

    # Record end time and calculate duration
    end_time = datetime.now()
    print(f"\nProgram ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {(end_time - start_time).total_seconds():.2f} seconds")

    # Save the result to a file
    output_file = "Code\\Step_3_Case_Languages\\CheckerDSL\\instance_2_extra_openai_10_new.txt"
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(instance_2)

    print(f"\nInstance 2 has been saved to '{output_file}'.")
    
    # Check for potential truncation
    if output_lines < instance_lines * 0.8:
        print("\n⚠️  WARNING: Output may be truncated (significantly fewer lines than input)")
    else:
        print("\n✓ Output appears complete")

if __name__ == "__main__":
    analyze_grammar_evolution_gpt()