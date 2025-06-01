# testing classifiying abililty
import asyncio
import time
from datetime import datetime
from dataclasses import dataclass
import json

from openai import AsyncOpenAI
from typing import List, Dict, Callable, Optional
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from gnosis import Classifier, Class


from datasets import load_dataset

ds = load_dataset("dair-ai/emotion", 
                  "split", 
                  split="train[300:600]", 
                  token=os.getenv("HF_TOKEN"))

# ds => {text: str, label: int}

# sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).
str2int = {
    "sadness": 0,
    "joy": 1,
    "love": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5,
}
int2str = {v: k for k, v in str2int.items()}

# tuning model on guessing right keyphrase, and make it transparent
classifier = Classifier([
    Class(name="SADNESS", values=["sad"]),
    Class(name="JOY", values=["joy"]),
    Class(name="LOVE", values=["love"]),
    Class(name="ANGER", values=["angry"]),
    Class(name="FEAR", values=["fear"]),
    Class(name="SURPRISE", values=["surprise"]),
])


@dataclass
class BatchSample:
    text: str
    label: int

@dataclass
class Result:
    text: str
    expected: int
    predicted: int
    correct: bool
    error: bool


client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

model = os.getenv("MODEL_NAME")


BASE_SYSTEM = """
You are an emotion classifier.
You will be given a text and you need to classify it into one of the following emotions:
- sadness
- joy
- love
- anger
- fear
- surprise
return only one name of chosen emotion.
"""

async def llm(text: str, system: str = BASE_SYSTEM) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content":  text}
        ],
        temperature=0.0
    )
    if response.choices[0].message.content is None:
        return ""
    return response.choices[0].message.content.strip()


CLASS_PROMPT = """
You will be given some class name, your task
is to yield a list of keywords or keyphrases that are most likely to be associated with that class.
Ensure that uniquness and importance of each keyword for the class, they must not contradict with other classes.
Yield in the format: keyword1; keyword2; keyword3; ...
"""


async def init_classifier() -> str:
    with open("classifier.json", "r") as f:
        data = json.load(f)
    classes = [Class(name=name.upper(), values=list(set(values))[:], fuzzy_temperature=0.0) 
               for name, values in data.items()]
    return Classifier(classes)

async def classify(text: str, expected: int, base=False) -> Result:
    expected = int2str[expected]
    error = False
    if base:
        predicted = await llm(text)
        if predicted is None:
            error = True
    else:
        predicted = classifier.repair(text)
    return Result(text=text, expected=expected, predicted=predicted, correct=predicted == expected, error=error)


async def collect_batch(batch: List[BatchSample], base=False) -> List[Result]:
    tasks = []
    for sample in batch:
        tasks.append(classify(sample.text, sample.label, base))
    results = await asyncio.gather(*tasks)
    return results

async def batch2json(batch: List[Result], batch_duration: float, base=False) -> List[dict]:
    return [{
        "text": result.text,
        "expected": result.expected,
        "predicted": result.predicted,
        "correct": result.correct,
        "error": result.error,
        "duration": batch_duration,
        "base": base
    } for result in batch if result.error is False]

async def main():
    global classifier
    classifier = await init_classifier()
    print("Created classifier with grammar:", classifier.grammar)
    start_time = time.time()
    total_samples = len(ds)
    print(f"🚀 Starting Gnosis benchmark on {total_samples} samples...")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    batch_size = 5
    total_batches = (total_samples + batch_size - 1) // batch_size
    all_results = []

    for i in range(0, total_samples, batch_size):
        batch_start_time = time.time()
        batch_end = min(i + batch_size, total_samples)
        batch_num = i // batch_size + 1

        # Load batch data on-demand
        batch = [
            BatchSample(
                text=ds["text"][idx],
                label=ds["label"][idx]
            )
            for idx in range(i, batch_end)
        ]

        print(f"🔄 Processing batch {batch_num}/{total_batches} (samples {i+1}-{batch_end})")

        grammar_batch_start_time = time.time()
        batch_results = await collect_batch(batch)
        grammar_batch_duration = time.time() - grammar_batch_start_time

        base_batch_start_time = time.time()
        base_batch_results = await collect_batch(batch, base=True)
        base_batch_duration = time.time() - base_batch_start_time

        batch_json = await batch2json(batch_results, grammar_batch_duration, base=False)
        base_batch_json = await batch2json(base_batch_results, base_batch_duration, base=True)
        all_results.extend(batch_json + base_batch_json)

        batch_duration = time.time() - batch_start_time

        print(f"🔄 Batch {batch_num}/{total_batches} completed in {batch_duration:.2f} seconds")

    # Final statistics
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print(f"Base accuracy: {len([r for r in all_results if r['correct'] and r['base']])/total_samples:.2%}")
    print(f"Grammar accuracy: {len([r for r in all_results if r['correct'] and not r['base']])/total_samples:.2%}")
    print("="*60)
    print(f"⏱️  Total time: {total_time:.2f}s")

    # Save results
    output_data = {
        "metadata": {
            "total_samples": total_samples,
            "total_time": total_time,
            "batch_size": batch_size,
            "total_batches": total_batches,
        },
        "results": all_results
    }
    
    output_file = "classify_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"✅ Results saved to {output_file}")
    
if __name__ == "__main__":
    asyncio.run(main())
    