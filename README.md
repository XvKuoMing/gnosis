# Gnosis: Grammar-Based Repair Framework

**Gnosis** is a powerful framework that utilizes grammar repair mechanisms for text generation and transformation. It implements a post-processing technique based on **LALR (Look-Ahead Left-to-Right)** parsing that helps transform free-text into any structured format based on context-free grammars.

## 🎯 Purpose

The core innovation of Gnosis lies in solving the task of **generation as repair** - transforming input text to desired output formats by repairing grammar violations rather than generating from scratch. This approach treats malformed or unstructured text as "broken" structured data that can be systematically repaired using grammar rules.

## 🔧 How It Works

### LALR-Based Post-Processing

Gnosis employs a sophisticated grammar repair mechanism built on LALR parsing:

1. **Grammar Definition**: Define your target structure using context-free grammar rules
2. **Repair Process**: When input text violates the grammar, the system uses beam search to find the minimal set of token insertions needed to make the text valid
3. **Reconstruction**: The repaired parse tree is reconstructed into the desired output format

### Core Components

- **`Rebuilder`**: The heart of the grammar repair mechanism using LALR interactive parsing
- **`Gnosis`**: Main framework class that sets up grammars and handles the repair process  
- **`Classifier`**: Specialized implementation for text classification tasks
- **`Class`**: Terminal definition for classification with fuzzy matching capabilities

## 🚀 Use Cases

### 1. Text Classification
Transform free-text into specific emotion categories:

```python
from gnosis import Classifier, Class

# Define emotion classes with keywords
classifier = Classifier([
    Class(name="SADNESS", values=["feeling.*?down", "feeling.*?depressed"]),
    Class(name="JOY", values=["happy", "excited", "cheerful"]),
    Class(name="ANGER", values=["angry", "furious", "irritated"]),
])

# Repair free text to structured classification
result = classifier.repair("I'm feeling really down today")
# Output: "sadness"
```

### 2. Function Call Generation

Transform natural language into structured function calls using custom grammars. Here's a complete example:

**Step 1: Define the Grammar using Lark syntax**
```lark
start: tools

tools: "[" tool+ "]"

tool: "{" "\"name\"" ":" "\"" FUNCTION_NAME "\"" "," "\"arguments\"" ":" arguments "}"

arguments: "{" argument ("," argument)* "}"
argument: "\"" ARGUMENT_NAME "\"" ":" ARGUMENT_VALUE


FUNCTION_NAME.3: "sum_of_multiples" | "product_of_primes"
ARGUMENT_NAME.2: "lower_limit" | "upper_limit" | "multiples" | "count"
ARGUMENT_VALUE.1: /\d+/

%import common.WS
%ignore WS
```

**Step 2: Create the Rebuilder**
```python
from rebuilder import Rebuilder

grammar = """
start: tools

tools: "[" tool+ "]"

tool: "{" "\"name\"" ":" "\"" FUNCTION_NAME "\"" "," "\"arguments\"" ":" arguments "}"

arguments: "{" argument ("," argument)* "}"
argument: "\"" ARGUMENT_NAME "\"" ":" ARGUMENT_VALUE


FUNCTION_NAME.3: "sum_of_multiples" | "product_of_primes"
ARGUMENT_NAME.2: "lower_limit" | "upper_limit" | "multiples" | "count"
ARGUMENT_VALUE.1: /\d+/

%import common.WS
%ignore WS
"""

rebuilder = Rebuilder(grammar)
```

**Step 3: Repair Free-form Input**
```python
# Input: Natural language or partial function call
input_text = 'Here is my function "sum_of_multiples" with arguments: lower_limit=1 and upper_limit=1000'

# Grammar-repaired output: Complete, valid function call
result = rebuilder.repair(input_text)
# Output: "[{"name": "sum_of_multiples", "arguments": {"lower_limit": 1, "upper_limit": 1000}}]"
```


### 3. Structured Data Extraction
Convert unstructured text into JSON, XML, or any context-free format by defining appropriate grammars.

### 4. Code Generation
Repair malformed code snippets to valid syntax in any programming language.

## ✅ Advantages

### 1. **Robustness**
- Handles malformed input gracefully through systematic repair
- No need for perfect input formatting

### 2. **Flexibility** 
- Any context-free grammar can be used
- Easily extensible to new domains and formats
- Supports fuzzy matching with configurable temperature

### 3. **Efficiency**
- LALR parsing provides linear time complexity
- Beam search optimization prevents exhaustive exploration
- Minimal token insertion strategy

### 4. **Interpretability**
- Clear grammar rules make the system explainable
- Repair process shows exactly what tokens were added
- No black-box transformations

### 5. **Consistency**
- Grammar constraints ensure valid output format
- Deterministic repair process for reproducible results

## 📊 Benchmark Results

The framework has been extensively tested on:

- **Emotion Classification**: Tested on 3000 samples from the `dair-ai/emotion` dataset
- **Function Calling**: Evaluated on 5000 samples from `Salesforce/xlam-function-calling-60k`

See `benchmark/test_class.py` and `benchmark/test_tools.py` for detailed test implementations and `benchmark/` directory for comprehensive analysis results.

## 🛠 Installation

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
gnosis/
├── gnosis.py              # Main framework and Classifier
├── rebuilder.py           # LALR-based repair mechanism  
├── benchmark/
│   ├── test_class.py      # Classification benchmarks
│   ├── test_tools.py      # Function calling benchmarks
│   └── *.json            # Benchmark results
└── requirements.txt       # Dependencies
```


