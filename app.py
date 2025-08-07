import asyncio
import aiohttp
from datasets import load_dataset
from tqdm import tqdm
import time
import tiktoken

import os
from dotenv import load_dotenv

# Charger le fichier .env
load_dotenv()

# Lire les variables d'environnement
API_URL = os.getenv("OPENAI_API_BASE")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_API_MODEL")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

MAX_FAILURE_RATE = 0.5  # stop if 50% fail
RATES_RPM = [30, 60, 120, 300, 600, 1200]  # escalating RPM tests
DURATION_SEC = 60

# === Tokenizer pour OpenAI ===
#tokenizer = tiktoken.encoding_for_model(MODEL)

#def count_tokens(text):
#    return len(tokenizer.encode(text))

# Explicitly get the encoding for the model
# "cl100k_base" is commonly used for gpt-3.5-turbo, but check your API docs if using another model.
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(encoding.encode(text))

# === Charger des prompts depuis SharedGPT ===
dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train[:200]")
prompts = [x["prompt"] for x in dataset if x.get("prompt")]

# === Fonction pour envoyer une requête ===
async def query(session, prompt):
    try:
        json_data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        start_time = time.time()
        async with session.post(API_URL, headers=HEADERS, json=json_data, timeout=30) as response:
            if response.status != 200:
                return False, 0, 0
            result = await response.json()
            content = result["choices"][0]["message"]["content"]
            prompt_tokens = count_tokens(prompt)
            completion_tokens = count_tokens(content)
            duration = time.time() - start_time
            return True, prompt_tokens + completion_tokens, duration
    except Exception:
        return False, 0, 0

# === Test à un taux donné ===
async def run_test(rpm):
    print(f"\n🔁 Testing {rpm} requêtes/minute ({rpm / 60:.2f} rps)")
    interval = 60 / rpm
    success_count = 0
    fail_count = 0
    total_tokens = 0
    total_duration = 0

    async with aiohttp.ClientSession() as session:
        tasks = []
        start = time.time()
        for i in range(rpm):
            prompt = prompts[i % len(prompts)]
            task = asyncio.create_task(query(session, prompt))
            tasks.append(task)
            await asyncio.sleep(interval)

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            ok, tokens, duration = await task
            if ok:
                success_count += 1
                total_tokens += tokens
                total_duration += duration
            else:
                fail_count += 1

    failure_rate = fail_count / (success_count + fail_count)
    tps = total_tokens / total_duration if total_duration > 0 else 0

    print(f"✅ Success: {success_count}, ❌ Fail: {fail_count}, 📉 Échecs: {failure_rate*100:.1f}%")
    print(f"⚡️ {tps:.2f} tokens/sec")

    return failure_rate < MAX_FAILURE_RATE

# === Exécution principale ===
async def main():
    for rpm in RATES_RPM:
        ok = await run_test(rpm)
        if not ok:
            print(f"\n🛑 Arrêt car plus de 50% des requêtes ont échoué à {rpm} RPM.")
            break

asyncio.run(main())
