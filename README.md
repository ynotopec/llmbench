# LLMBench – Benchmarker de modèles LLM via API OpenAI-compatible

Ce script Python permet de benchmarker des modèles LLM compatibles avec l'API OpenAI (v1/chat/completions), en mesurant :
- le taux de réussite,
- la latence moyenne,
- le débit en tokens/seconde.

Il utilise des prompts de qualité extraits de [Awesome ChatGPT Prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts).

## 🚀 Prérequis

- Python 3.9+
- Accès à une API OpenAI-compatible (e.g. vLLM, LiteLLM, Mistral, etc.)

## 🧪 Installation

```bash
git clone https://github.com/<ton-utilisateur>/llmbench.git
cd llmbench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

## ⚙️ Configuration

Édite les variables `API_URL`, `API_KEY`, et `MODEL` dans `app.py` pour tester un modèle spécifique.

## ▶️ Utilisation

```bash
python app.py
```

## 📦 Dataset utilisé

[Awesome ChatGPT Prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts)

## 📄 Licence

MIT
