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
short_description: 'CLAPP (qiskit ver.): qiskit LLM Agent for Pair Programming'
license: mit
---

# CLAPP (qiskit ver.) - qiskit LLM Agent for Pair Programming

<p align="center">
  <img src="images/CLAPP.png" alt="CLAPP Logo" width="400"/>
</p>

CLAPP (qiskit ver.) is a Streamlit application that provides an AI pair programming assistant specialized in the [qiskit](https://qiskit.org/) quantum computing code. It uses LangChain and OpenAI models, leveraging Retrieval-Augmented Generation (RAG) with qiskit documentation and code examples to provide informed responses and assist with coding tasks.

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

This project uses conda/mamba for environment management, which is compatible with qiskit installation requirements.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/daimurat/QAMP2025.git
   cd 02_multi_agents
   ```

2. **Create a conda environment from the environment.yml file:**
   ```bash
   # Using conda
   conda env create -f environment.yml
   
   # Or using mamba (faster)
   mamba env create -f environment.yml
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

## Project Structure

* `CLAPP.py`: The main Streamlit application script.
* `install_classy.sh`: Script to install CLASS from source.
* `test_classy.py`: Script to test CLASS installation and functionality.
* `environment.yml`: Conda environment specification with all required dependencies.
* `class-data/`: Directory containing data for the RAG system (CLASS code, docs, etc.).
* `prompts/`: Directory containing system prompts for the AI agents.
* `images/`: Contains images used in the app interface, including the CLAPP logo.
* `{username}_encrypted_api_key`: Stores the encrypted OpenAI API keys for each user.
