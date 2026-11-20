# QDBBQ: Evolving Diverse Test Cases for Bias Evaluation in Large Language Models

> Official implementation of **"QDBBQ: Evolving Diverse Test Cases for Bias Evaluation in Large Language Models"**  


## Setup

**1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)**
```bash
pip install uv
```

**2. Install dependencies**
```bash
uv sync
```

**3. Set your LLM API**

We use the Gemma-4-31B as our variation LLM, which require API keys. 

Paste your OpenRouter API key to the api_key attribute of MutationModelConfig class in the [`openelm/configs.py`](openelm/configs.py) file.


## Experiments

### Run the main method and baselines 

To run the main QDBBQ method:
```bash
uv run main.py run_name=<run_name>
```

**Example:**
```bash
uv run main.py run_name=logs/methods/qdbbq/
```

To run the zero-shot sampling baseline:
```bash
run python main.py run_name=<run_name> qd.no_variation=True
```

To run the current niche only baseline:
```bash
uv run main.py run_name=<run_name> qd.current_niche_only=True
```


### Run the bias discovery on different LLMs

```bash
uv run main.py run_name=<run_name> target.model_path=<model_card_name>
```

**Example:**
```bash
uv run main.py run_name=logs/llms/qwen-9b/ target.model_path=Qwen/Qwen3.5-9B
```


For a full list of hyperparameters, see [`openelm/configs.py`](openelm/configs.py).


## Acknowledgements

This codebase builds on:

- **[OpenELM](https://github.com/CarperAI/OpenELM.git)** by CarperAI


## Citation

```bibtex
@inproceedings{QDBBQ2026,
  author    = {Thai Huy Nguyen, Quan Minh Phan, Khang Tran, Van Bich Nguyen, Tien-Dung Mai, Vo Nguyen Le Duy, and Ngoc Hoang Luong},
  title     = {{QDBBQ: Evolving Diverse Test Cases for Bias Evaluation in Large Language Models}},
  year      = {2026}
}
```