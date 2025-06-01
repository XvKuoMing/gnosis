import asyncio
from datasets import load_dataset
import json
from jinja2 import Template
from openai import AsyncOpenAI
from dataclasses import dataclass
from typing import Optional, List
import time
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rebuilder import Rebuilder
from dotenv import load_dotenv

load_dotenv()

ds = load_dataset("Salesforce/xlam-function-calling-60k", 
                  split="train[0:5000]", 
                  token=os.getenv("HF_TOKEN"))

# ds_filtered = ds.filter(lambda x: len(json.loads(x["answers"])) == 1)

def keep_simple(example):
    """
    Filter function to check if all tools in the example have parameters
    that are only of type 'str' or 'int'.
    """
    try:
        if len(json.loads(example["answers"])) != 1:
            return False
        tools = json.loads(example["tools"])
        
        for tool in tools:
            parameters = tool["parameters"]
            for _, param_info in parameters.items():
                param_type = param_info["type"]
                # Check if the type is not str or int
                if param_type not in ["str", "int"]:
                    return False
        return True
    except (json.JSONDecodeError, KeyError, TypeError):
        # If there's any error parsing, exclude the example
        return False

# Apply the additional filter for str/int parameters only
ds = ds.filter(keep_simple)


client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
model = os.getenv("MODEL_NAME")

@dataclass(frozen=True)
class Result:
    tools: str
    query: str
    chosen_tool: str
    expected_tool: str
    syntactic_validity: bool
    semantic_validity: bool
    grammar_enabled: bool
    error: bool = False


def format_prompt(query: str, tools: str) -> str:
    return f"""<tools>{tools}</tools><query>{query}</query>"""


SYSTEM_BASE = """
You are given a text and tools
your task is to choose the correct tool for this query.

Here is example:
```
<query>
Find the sum of all the multiples of 3 and 5 between 1 and 1000. Also find the product of the first five prime numbers.
</query>
<tools>
[
    {
      "name": "math_toolkit.sum_of_multiples",
      "description": "Find the sum of all multiples of specified numbers within a specified range.",
      "parameters": {
        "lower_limit": {
          "type": "int",
          "description": "The start of the range (inclusive).",
          "required": true
        },
        "upper_limit": {
          "type": "int",
          "description": "The end of the range (inclusive).",
          "required": true
        },
        "multiples": {
          "type": "list",
          "description": "The numbers to find multiples of.",
          "required": true
        }
      }
    },
    {
      "name": "math_toolkit.product_of_primes",
      "description": "Find the product of the first n prime numbers.",
      "parameters": {
        "count": {
          "type": "int",
          "description": "The number of prime numbers to multiply together.",
          "required": true
        }
      }
    }
  ],
</tools>
<output>
[
    {
      "name": "math_toolkit.sum_of_multiples",
      "arguments": {
        "lower_limit": 1,
        "upper_limit": 1000,
        "multiples": [3, 5]
      }
    },
    {
      "name": "math_toolkit.product_of_primes",
      "arguments": {
        "count": 5
      }
    }
]
</output>
```
Return only the output as in the example, nothing else. No other text, comments.
Ensure the output is valid json and could be parsed by json.loads.
"""

SYSTEM_GRAMMAR = """
You are given a text and tools
your task is to choose the correct tool for this query.
Here is an example:
```
<query>
query here
</query>
<tools>
tools here
</tools>

```
Output the result in free form, but correct order:
name of function 
then argument name in the format $argument_name
then the value of the argument
the argument value must correspond to the function value description and must exactly match expected value

Example:
```
name_of_function $argument_name argument_value here
like this:
sum_of_multiples $lower_limit 1 $upper_limit 1000
```
you can output only one expected function
"""


async def llm(user_prompt: str, system_prompt: str) -> Optional[str]:
    
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    message = response.choices[0].message
    content = message.content
    # if using reasoning type model
    # if content is None and hasattr(message, "reasoning_content"):
    #     content = message.reasoning_content.strip() # try to find from reasoning
    if content is None:
        return None
    content = content.replace("```json", "").replace("```", "")
    content = content.replace("\n", "").strip()
    return content


async def create_grammar(tools: str) -> str:
    tools = json.loads(tools)
    functions: List[str] = []
    arguments_names: List[str] = []
    for tool in tools:
        functions.append(tool["name"])
        arguments_names += list(tool["parameters"].keys())
    functions = "|".join(functions)
    arguments_names = "|".join(arguments_names)
    template = Template(r"""
start: tools

tools: "[" tool+ "]"

tool: "{" "\"name\"" ":" "\"" FUNCTION_NAME "\"" "," "\"arguments\"" ":" arguments "}"

arguments: "{" argument ("," argument)* "}"
argument: "\"" ARGUMENT_NAME "\"" ":" "\"" ARGUMENT_VALUE "\""

FUNCTION_NAME.3: /({{ functions }})/
ARGUMENT_NAME.2: /\$(?:{{ arguments_names }})/
ARGUMENT_VALUE.1: /[\w\d,\s-:_\/\.]+/
                        
%import common.WS
%ignore WS
""")
    return template.render(functions=functions, arguments_names=arguments_names)


missing_token = lambda tkn, defn, is_regex: tkn if is_regex else defn

async def repair_with_grammar(content: str, grammar: str) -> str:
    rebuilder = Rebuilder(
        grammar=grammar,
        token_transformer=missing_token
    )
    repaired = rebuilder.repair(content, beam_width=25, strategy="shortest", break_early_limit=7)
    repaired = repaired.replace("$", "")
    return repaired


async def check_syntactic_validity(content: str) -> bool:
    try:
        json.loads(content)
        return True
    except json.JSONDecodeError:
        return False


async def check_semantic_validity(content: str, tools: str) -> bool:
    """
    soft check for semantic validity
    """
    try:
        generated_tool = json.loads(content)[0] # it must be json with only one tool by conditions of filtering
        expected_tool = json.loads(tools)[0]
        if generated_tool["name"] != expected_tool["name"]:
            return False
        for argument_name, argument_value in expected_tool["arguments"].items():
            if argument_name not in generated_tool["arguments"]:
                return False
            if str(generated_tool["arguments"][argument_name]).strip() != str(argument_value).strip():
                return False
        return True
    except (json.JSONDecodeError, IndexError, KeyError, TypeError):
        return False



async def choose_tool(query: str, tools: str, expected_tool: str, enable_grammar: bool = False) -> Result:
    if enable_grammar:
        system_prompt = SYSTEM_GRAMMAR
    else:
        system_prompt = SYSTEM_BASE
    query = format_prompt(query, tools)
    content = await llm(query, system_prompt)
    if content is None:
        return Result(
            tools=tools, 
            query=query, 
            chosen_tool="", 
            expected_tool=expected_tool,
            grammar_enabled=enable_grammar,
            syntactic_validity=False, 
            semantic_validity=False, 
            error=True)
    
    if enable_grammar:
        grammar = await create_grammar(tools)
        content = await repair_with_grammar(content, grammar)

    syntactic_validity = await check_syntactic_validity(content)
    if syntactic_validity:
        semantic_validity = await check_semantic_validity(content, expected_tool)
    else:
        semantic_validity = False

    return Result(
        tools=tools, 
        query=query, 
        chosen_tool=content, 
        expected_tool=expected_tool,
        grammar_enabled=enable_grammar,
        syntactic_validity=syntactic_validity, 
        semantic_validity=semantic_validity
        )


@dataclass
class BatchSample:
    query: str
    tools: str
    expected_tool: str

async def collect_batch(batch: List[BatchSample], enable_grammar: bool = False) -> List[Result]:
    tasks = []
    for sample in batch:
        tasks.append(choose_tool(sample.query, sample.tools, sample.expected_tool, enable_grammar))
    results = await asyncio.gather(*tasks)
    return results


async def batch2json(batch: List[Result], batch_duration: float) -> List[dict]:
    return [{
        "query": result.query,
        "tools": result.tools,
        "expected_tool": result.expected_tool,
        "chosen_tool": result.chosen_tool,
        "grammar_enabled": result.grammar_enabled,
        "syntactic_validity": result.syntactic_validity,
        "semantic_validity": result.semantic_validity,
        "duration": batch_duration
    } for result in batch if result.error is False]

async def main():
    start_time = time.time()
    total_samples = len(ds)
    print(f"🚀 Starting Gnosis benchmark on {total_samples} samples...")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    batch_size = 5
    total_batches = (total_samples + batch_size - 1) // batch_size
    all_results = []
    
    # Track overall statistics
    total_syntactic_base = 0
    total_semantic_base = 0
    total_syntactic_grammar = 0
    total_semantic_grammar = 0
    total_processed = 0
    total_base_time = 0
    total_grammar_time = 0
    
    for i in range(0, total_samples, batch_size):
        batch_start_time = time.time()
        batch_end = min(i + batch_size, total_samples)
        batch_num = i // batch_size + 1

        # Load batch data on-demand
        batch = [
            BatchSample(
                query=ds["query"][idx],
                tools=ds["tools"][idx],
                expected_tool=ds["answers"][idx]
            )
            for idx in range(i, batch_end)
        ]

        print(f"🔄 Processing batch {batch_num}/{total_batches} (samples {i+1}-{batch_end})")

        # Process base model
        batch_base_start = time.time()
        batch_results_base: List[Result] = await collect_batch(batch, enable_grammar=False)
        batch_base_duration = time.time() - batch_base_start
        
        # Process grammar model
        batch_grammar_start = time.time()
        batch_results_grammar: List[Result] = await collect_batch(batch, enable_grammar=True)
        batch_grammar_duration = time.time() - batch_grammar_start
        
        batch_end_time = time.time()
        batch_total_duration = batch_end_time - batch_start_time

        # Convert results to JSON format and add to all_results
        batch_json_base = await batch2json(batch_results_base, batch_base_duration)
        batch_json_grammar = await batch2json(batch_results_grammar, batch_grammar_duration)
        
        # Parse and add to all_results
        all_results.extend(batch_json_base)
        all_results.extend(batch_json_grammar)

        # Calculate batch statistics
        batch_syntactic_base = sum(1 for r in batch_results_base if r.syntactic_validity and not r.error)
        batch_semantic_base = sum(1 for r in batch_results_base if r.semantic_validity and not r.error)
        batch_syntactic_grammar = sum(1 for r in batch_results_grammar if r.syntactic_validity and not r.error)
        batch_semantic_grammar = sum(1 for r in batch_results_grammar if r.semantic_validity and not r.error)
        
        valid_base_count = len([r for r in batch_results_base if not r.error])
        valid_grammar_count = len([r for r in batch_results_grammar if not r.error])
        
        # Update totals
        total_syntactic_base += batch_syntactic_base
        total_semantic_base += batch_semantic_base
        total_syntactic_grammar += batch_syntactic_grammar
        total_semantic_grammar += batch_semantic_grammar
        total_processed += len(batch)
        total_base_time += batch_base_duration
        total_grammar_time += batch_grammar_duration
        
        print(f"   ✅ Batch completed in {batch_total_duration:.2f}s")
        print(f"   ⏱️  Base model: {batch_base_duration:.2f}s | Grammar model: {batch_grammar_duration:.2f}s")
        print(f"   📈 Base model - Syntactic: {batch_syntactic_base}/{valid_base_count} ({batch_syntactic_base/valid_base_count*100:.1f}%), Semantic: {batch_semantic_base}/{valid_base_count} ({batch_semantic_base/(valid_base_count)*100:.1f}%)")
        print(f"   📈 Grammar model - Syntactic: {batch_syntactic_grammar}/{valid_grammar_count} ({batch_syntactic_grammar/valid_grammar_count*100:.1f}%), Semantic: {batch_semantic_grammar}/{valid_grammar_count} ({batch_semantic_grammar/(valid_grammar_count)*100:.1f}%)")
        
        # Show progress
        progress = (total_processed / total_samples) * 100
        elapsed_time = time.time() - start_time
        estimated_total = elapsed_time / progress * 100 if progress > 0 else 0
        remaining_time = estimated_total - elapsed_time
        
        print(f"   🎯 Progress: {progress:.1f}% | Elapsed: {elapsed_time:.1f}s | ETA: {remaining_time:.1f}s")
        print("   " + "-" * 50)
    
    # Final statistics
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⏱️  Base model total time: {total_base_time:.2f}s (avg: {total_base_time/total_batches:.2f}s per batch)")
    print(f"⏱️  Grammar model total time: {total_grammar_time:.2f}s (avg: {total_grammar_time/total_batches:.2f}s per batch)")
    print(f"📝 Total samples processed: {total_processed}")
    print(f"📈 Base model - Syntactic: {total_syntactic_base}/{total_processed} ({total_syntactic_base/total_processed*100:.1f}%), Semantic: {total_semantic_base}/{total_processed} ({total_semantic_base/total_processed*100:.1f}%)")
    print(f"📈 Grammar model - Syntactic: {total_syntactic_grammar}/{total_processed} ({total_syntactic_grammar/total_processed*100:.1f}%), Semantic: {total_semantic_grammar}/{total_processed} ({total_semantic_grammar/total_processed*100:.1f}%)")
    
    # Save results
    output_data = {
        "metadata": {
            "total_samples": total_processed,
            "total_time": total_time,
            "base_model_time": total_base_time,
            "grammar_model_time": total_grammar_time,
            "avg_base_time_per_batch": total_base_time / total_batches,
            "avg_grammar_time_per_batch": total_grammar_time / total_batches,
            "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "model": model,
            "batch_size": batch_size,
            "total_batches": total_batches
        },
        "summary": {
            "base_model": {
                "syntactic_accuracy": total_syntactic_base / total_processed,
                "semantic_accuracy": total_semantic_base / total_processed
            },
            "grammar_model": {
                "syntactic_accuracy": total_syntactic_grammar / total_processed,
                "semantic_accuracy": total_semantic_grammar / total_processed
            }
        },
        "results": all_results
    }
    
    with open("benchmark_gnosis_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to benchmark_gnosis_results.json")


if __name__ == "__main__":
    asyncio.run(main())
