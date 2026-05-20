import os
from anthropic import Anthropic
from datetime import datetime

def read_file(filename):
    """Read content from a file"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        return None

def get_response_content_stream(stream):
    """Extract content from streaming response"""
    full_text = ""
    try:
        for text in stream.text_stream:
            full_text += text
        return full_text
    except Exception as e:
        print(f"Warning: Error processing stream content: {e}")
        return full_text

def get_max_tokens_for_instance(instance_content):
    """Dynamically set max_tokens based on instance size"""
    if instance_content is None:
        return 4096
    
    lines = len(instance_content.split('\n'))
    print(f"Instance has {lines} lines")
    
    if lines < 100:
        max_tokens = 4096
    elif lines < 200:
        max_tokens = 8192
    elif lines < 500:
        max_tokens = 16384
    elif lines < 1000:
        max_tokens = 32768
    else:
        max_tokens = 65536  # Close to Claude Sonnet 4.5's limit
    
    print(f"Setting max_tokens={max_tokens}")
    return max_tokens

def analyze_grammar_evolution(api_key):
    """Main function: Analyze grammar evolution using Chain-of-Thought approach with streaming"""
    # Record start time
    start_time = datetime.now()
    print(f"Program started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Read files
    grammar1 = read_file('Code\\Step_3_Case_Languages\\CheckerDSL\\grammar_1_20150503_55911bf.txt')
    grammar2 = read_file('Code\\Step_3_Case_Languages\\CheckerDSL\\grammar_2_20150727_3fa6e6d.txt')
    instance1 = read_file('Code\\Step_3_Case_Languages\\CheckerDSL\\instance_1_20250503_55911bf.txt')
    
    if None in [grammar1, grammar2, instance1]:
        print("File reading failed. Please check file paths and contents.")
        return
    
    # Dynamically set max_tokens for the final step
    final_max_tokens = 64000
    
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)
    messages = []
    final_result = []
    
    print("\n=== Step 1: Sending Grammar 1 ===")
    # Step 1: Introduce grammar_1
    messages.append({
        "role": "user",
        "content": f"""
        I will show you two versions of a grammar and an instance that conforms to the first grammar. 
        Let's start with the first grammar version:

        {grammar1}

        Please remember the content.
        """
    })
    
    with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1000,
        messages=messages
    ) as stream:
        content = get_response_content_stream(stream)
    
    messages.append({"role": "assistant", "content": content})
    print("Grammar 1 acknowledged")
    
    print("\n=== Step 2: Sending Grammar 2 ===")
    # Step 2: Introduce grammar_2
    messages.append({
        "role": "user",
        "content": f"""
        Now, here's the second version of the grammar:

        {grammar2}

        Please remember the content also.
        """
    })
    
    with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1000,
        messages=messages
    ) as stream:
        content = get_response_content_stream(stream)
    
    messages.append({"role": "assistant", "content": content})
    print("Grammar 2 acknowledged")
    
    print("\n=== Step 3: Processing Instance (this may take several minutes) ===")
    # Step 3: Process instance with streaming
    messages.append({
        "role": "user",
        "content": f"""
        Now, here's an instance that conforms to the first grammar:

        {instance1}

        grammar_1 is the initial grammar of the DSL. We evolved it to get grammar_2. instance_1 was originally a text instance that followed grammar_1. Now I want you to analyze the differences between the two versions of the grammar, and based on this difference, modify instance_1 and get instance_2, which will follow grammar_2. Please address the following things:
        1. When evolving the instance, please do not omit any mandatory elements, such as characters enclosed by single quotes
        2. If grammar_2 adds a new grammar rule or a new attribute that is optional or in an "OR" relationship (i.e., |), then please do not instantiate it.
        3. Do not miss or add any auxiliary information in the instance, e.g., comments, formats (white space, indents, tabs, empty lines, etc.).
        4. When evolving the instance, pay special attention to grammar rules that have been deleted in the evolved grammar. If a grammar rule no longer exists in Grammar 2, you must remove all instances of that rule from the evolved instance.
        5. Always output your best attempt at the evolved instance, even if incomplete. Never return an empty result.
        """
    })
    
    # Use streaming for the final transformation
    step3_start = datetime.now()
    with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=final_max_tokens,
        messages=messages
    ) as stream:
        content = get_response_content_stream(stream)
        
        # Get actual token usage
        final_message = stream.get_final_message()
        actual_output_tokens = final_message.usage.output_tokens
        print(f"\nActually used {actual_output_tokens} output tokens (limit was {final_max_tokens})")
    
    step3_end = datetime.now()
    step3_duration = (step3_end - step3_start).total_seconds()
    print(f"Step 3 completed in {step3_duration:.2f} seconds")
    
    final_result.append("=== Step 3: Instance Transformation ===\n" + content)
    final_text = '\n'.join(final_result)

    # Record end time and calculate duration
    end_time = datetime.now()
    print(f"\nProgram ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = end_time - start_time
    print(f"Total execution time: {duration.total_seconds():.2f} seconds")
    
    # Save the final result
    output_file = 'Code\\Step_3_Case_Languages\\CheckerDSL\\instance_2_extra_claude_10.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(str(final_text))
    
    print(f"Analysis completed. The result has been saved in {output_file}")
    print(f"Output file size: {len(final_text)} characters")
    
if __name__ == "__main__":
    # Input API key    
    analyze_grammar_evolution('')