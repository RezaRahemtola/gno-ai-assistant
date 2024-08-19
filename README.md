# Gno AI assistant

PoC of AI assistant in the form of a chatbot for the Gno ecosystem.
It's fed with the Markdown content of the documentation.
On each message, a RAG-like process is done to detect relevant document chunks based on generated embedding vectors.

## ⚙️ Installation

### AI

Setup your Python environment with these commands:
```shell
cd ai/
python -m venv venv
python -m pip install poetry
poetry install
```

Then you can create a `.env` file and fill it with values inspired from the [`.env.example`](/ai/.env.example) file.


### Front

Setup your frontend with these commands:
```shell
cd front/
yarn
```

## 🚀 Getting started

Now you're only two command away from using your Gno AI assistant 🔥

Start the AI
```shell
python src/main.py
```

> 💡 It might be slow to start due to the generation of embedding vectors for the document chunks.

and launch the frontend
```shell
yarn dev
```



<div align="center">
  <h2>Made with ❤️ by</h2>
  <a href="https://github.com/RezaRahemtola">
    <img src="https://github.com/RezaRahemtola.png?size=85" width=85/>
    <br>
    <sub>Reza Rahemtola</sub>
  </a>
</div>
