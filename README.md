---
title: CLAPP (qiskit ver.)
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: 'qiskit LLM Agent for Pair Programming'
license: mit
---

# QAMP2025
## Description
This project aims to design and evaluate multi-agent systems (MAS) that leverage Large Language Model (LLM) agents, enhanced with advanced prompt-engineering methods such as Retrieval-Augmented Generation (RAG), to improve quantum code generation. 
Standard LLMs are not trained on specialized scientific software and may hallucinate details, miss domain-specific context, or rely on outdated versions of libraries present in their training data. 
The envisioned tool would provide a conversational interface that fully understands the structure and features of the latest Qiskit releases, helping users with technical questions, code development and execution, debugging, and visualization.


# qiskit LLM Agent for Pair Programming

qiskit LLM Agent is a Streamlit application that provides an AI pair programming assistant specialized in the [qiskit](https://qiskit.org/) quantum computing code. It uses LangChain and OpenAI models, leveraging Retrieval-Augmented Generation (RAG) with qiskit documentation and code examples to provide informed responses and assist with coding tasks.

## Collaborators

* Daiki Murata
* Seemanta Bhattacharjee
* Francisco Villaescusa-Navarro
* Urbano L Franca
* inspired by the [CLAPP](https://github.com/santiagocasas/clapp)

## Features

* **Conversational AI:** Interact with an AI assistant knowledgeable about [qiskit](https://qiskit.org/).
* **Code Execution:** Executes Python code snippets in real-time, with automatic error detection and correction.
* **RAG Integration:** Retrieves relevant information from qiskit documentation and code (`./qiskit-data/`) to answer questions accurately.
* **Multiple Response Modes:** 
  * **Fast Mode:** Quick responses with good quality (recommended for most uses)
  * **Swarm Mode:** Multi-agent refined responses for more complex questions (takes longer)
* **Secure User Management:** Username-based API key storage allows multiple users to securely save encrypted API keys.
* **Real-time Feedback:** Streams installation and execution progress in real-time.
* **Model Selection:** Choose between different OpenAI models (GPT-4o, GPT-4o-mini).

## Setup and Installation

This project uses conda for environment management, which is compatible with qiskit installation requirements.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/daimurat/QAMP2025.git
   ```

2. **Create a conda environment from the environment.yml file:**
   ```bash
   conda env create -f environment.yml
   ```

   (Option) When you evaluate the agent by QiskitHumanEval, you need to another conda environment.
   ```bash
   conda env create -f environment_eval.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate streamapp
   ```

4. **API Key:**
   * You will need an OpenAI API or Gemini key.
   * The application allows you to enter a username, API key, and password to encrypt and store it locally.
   * Keys are saved as `{username}_encrypted_api_key` to allow multiple users.
   * Get your free Gemini key from https://aistudio.google.com/app/apikey

5. **environment variables file:**
   * You need to create a `.env` file in the root directory of the project.

   ```.env
   OPENAI_API_KEY=<your-openai-api-key>
   ```

6. **qiskit Data:**
   * Ensure the `qiskit-data` directory contains the necessary qiskit documentation for the RAG system.

7. **System Prompts:**
   * Ensure the `prompts/` directory contains the necessary instruction files (`qiskit_instructions.txt`, `review_instructions.txt`, etc.).

## Usage

### Web application 
1. **Activate the conda environment:**
   ```bash
   conda activate streamapp
   ```

2. **Run the Streamlit application:**
   ```bash
   streamlit run CLAPP.py
   ```

3. **Setup process:**
   * Enter your OpenAI API key and optionally a username and password for encryption.
   * Initialize the application by clicking "Initialize with Selected Model".
   * Start chatting with the assistant about qiskit-related questions.

4. **Code execution:**
   * When the assistant provides code, you can execute it by typing "execute!" in the chat.
   * The system will run the code, display the output, and show any generated plots.
   * If errors occur, the system will automatically attempt to fix them.

### Qiskit Dataset Evaluation
1. **Activate the conda environment:**
   ```bash
   conda activate qiskitagent
   ```

2. **Run the evaluation script:**
   ```bash
   python evaluate_qiskit_humaneval.py --max-items 10
   # or single-turn, no-RAG fast mode
   python evaluate_qiskit_humaneval.py --mode fast --no-rag --max-items 3
   ```

    | Option | Default | Description |
    |-----------|----------|------|
    | `--model` | `meta-llama/llama-4-maverick-17b-128e-instruct` (env: `GROQ_EVAL_MODEL`) | model name |
    | `--dataset` | `Qiskit/qiskit_humaneval` | dataset name |
    | `--split` | `test` | dataset split |
    | `--max-items` | `None` | maximum items to evaluate |
    | `--mode` | `deep` | `deep` (multi-agent) or `fast` (single-turn, no-RAG) |
    | `--temperature` | `0.2` | sampling temperature |
    | `--max-output-tokens` | `800` | maximum output tokens |
    | `--timeout-sec` | `45` | test execution timeout（sec） |
    | `--use-rag` / `--no-rag` | `True` | enable/disable RAG (ignored in `fast` mode) |
    | `--rag-top-k` | `5` | retrieval top_k |
    | `--rag-min-score` | `0.0` | minimum similarity score |
    | `--db-path` | `QAMP/data/qamp.db` | SQLite vector DB |
    | `--task-id` | `None` | run a specific task id |
    | `--difficulty` | `None` | filter by difficulty (basic/intermediate/difficult) |
    | `--outdir` | `out` | output directory |
    | `--dry-run` | `False` | skip inference |

## Project Structure

* `CLAPP.py`: The main Streamlit application script.
* `environment.yml`: Conda environment specification with all required dependencies.
* `class-data/`: Directory containing data for the RAG system (CLASS code, docs, etc.).
* `prompts/`: Directory containing system prompts for the AI agents.
* `images/`: Contains images used in the app interface, including the CLAPP logo.
* `{username}_encrypted_api_key`: Stores the encrypted OpenAI API keys for each user.
