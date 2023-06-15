# LangChain Router Chain Playground

This simple project shows how you can use *LLMRouterChain* to simulate different roles 
which are selected dynamically during your interaction with the LLM depending on your question.

It also contains a good example of a file based callback handler which you can use to save content to a text or HTML file.

## Conda install commands

```bash
conda create --name langchain python=3.10
conda install -c conda-forge openai
conda install -c conda-forge langchain
conda install -c https://conda.anaconda.org/conda-forge prompt_toolkit
```

## Execution

```bash
python ./lang_chain_router_chain.py
```