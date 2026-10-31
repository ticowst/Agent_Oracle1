"""
==========================================================
LangChain for AI Agents — Companion Code
==========================================================
Lesson 1: Core Building Blocks
  - Models, Prompts, Chains, Output Parsers, Memory, Tools

Demo order (matches the lessons):
  Lesson 1 → Models, Prompts, Chains, Memory
  Lesson 2 → Tools

Prerequisites:
  pip install langchain langchain-openai python-dotenv

Setup:
  Create a .env file with: OPENAI_API_KEY=sk-your-key-here
==========================================================
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
from pathlib import Path
from dotenv import load_dotenv

# Find the .env file sitting next to THIS script (not wherever we ran from).
# Path(__file__) = this file; .parent = its folder. This makes the key load
# correctly no matter which directory you launch python from.

script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env"

load_dotenv(dotenv_path=env_path)
print("Current working directory:", os.getcwd())
print("Script directory:", script_dir)
print("Using .env from:", env_path)

# Quick sanity check that the key actually loaded before we call the model.
print("OPENAI_API_KEY found:", bool(os.getenv("OPENAI_API_KEY")))

# ─────────────────────────────────────────────
# 1. MODELS — The Reasoning Engine
# ─────────────────────────────────────────────
# The "model" is the LLM itself — the part that actually thinks.
# init_chat_model gives ONE interface to every provider: to switch from
# OpenAI to Anthropic or Google, you change only the string below.

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini")

"""# The simplest possible use: send text in, get an answer back.
# .invoke() is the universal "run it" method across LangChain.
response = model.invoke("Que es LangChain en una sola oración")
print("=== Model Response ===")
print(response.content)
print()
"""
# ─────────────────────────────────────────────
# 2. PROMPT TEMPLATES — Steering the Model
# ─────────────────────────────────────────────
# A prompt template is a reusable sentence with blanks ({placeholders})
# you fill in later — write the wording once, reuse it many times.

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# PromptTemplate = a single plain-text string with blanks.
simple_template = PromptTemplate(
    input_variables = ["concept"],
    template = "Explica el {concept} en tono principiante en 2 o 3 oraciones."
)

# .format() fills the blank and returns the finished text.
formatted = simple_template.format(concept="AI Agents")
print("=== Formatted Prompt===")
print(formatted)
print()

# ChatPromptTemplate = built from ROLES (system / human), which is how chat
# models expect their input. "system" sets behavior, "human" is the user turn.
chat_template = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en IA que explica conceptos complejos de manera simple."),
    ("human", "Explícame {concept} con un ejemplo simple en Python.")
])

# .format_messages() fills the blanks and returns a LIST of messages.
messages = chat_template.format_messages(concept="Transformers")
# {concept} in the template is the empty blank, and concept="list comprehension" is you handing it the word that goes in the blank.
"""print("=== Chat Message ===")
for msg in messages:
    print(f"[{msg.type}]: {msg.content[:80]}...")
print()"""

# ─────────────────────────────────────────────
# 3. CHAINS — Connecting the Pieces with LCEL
# ─────────────────────────────────────────────
# The pipe | glues components into a pipeline. Each step's output flows
# into the next, left to right:  prompt | model | parser.

from langchain_core.output_parsers import StrOutputParser

# prompt fills the blanks → model answers → parser pulls out clean text.
# StrOutputParser just extracts the plain string from the model's reply
# object, so you don't have to write .content yourself every time.

chain = chat_template | model | StrOutputParser()

# Run the whole pipeline with a single .invoke().
result = chain.invoke({"concept": "LLMOps"})
print("=== Chain Output ===")
print(result)
print()

# ─────────────────────────────────────────────
# 4. MEMORY — Giving the Model Context
# ─────────────────────────────────────────────
# Models are stateless — they forget everything between calls.
# "Memory" is simply us storing past messages and feeding them back in.
# (Modern LangChain uses ChatMessageHistory; the old
#  ConversationBufferMemory is deprecated and out of the core package.)

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# A simple in-memory store that holds the conversation.
memory = InMemoryChatMessageHistory()

# Hand-build a short conversation so we have something to "remember".
memory.add_message(HumanMessage(content="Mi nombre es Francisco y estoy aprendiendo Agentic AI"))
memory.add_message(AIMessage(content="¡Hola Francisco! Agentic AI es una buena elección"))
memory.add_message(HumanMessage(content="Que herramientas debo aprender primero"))
memory.add_message(AIMessage(content="Te recomiendo empezar con PromptTemplates y chains, luego pasar aa tools y agentes"))

print("=== Memory Contents ===")
for msg in memory.messages:
    print(f" [{msg.type}]: {msg.content[:80]}...")
print()

# The "placeholder" slot is where the stored messages get injected into the
# prompt, so the model can SEE the earlier conversation.

chat_with_memory = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en IA. Usa la conversación histórica para personalizar tus respuestas"),
    ("placeholder","{history}"),
    ("human","{question}"),
])


chain_with_memory = chat_with_memory | model | StrOutputParser()

# We pass the stored history in alongside the new question.
# Watch the model correctly recall the name "Alex" — that's "memory".

result = chain_with_memory.invoke({
    "history": memory.messages,
    "question": "Cual es mi nombre"
})

print("=== Memory-Aware Response ===")
print(result)
print()

# ─────────────────────────────────────────────
# 5. TOOLS — Giving the Model Abilities   (Lesson 2)
# ─────────────────────────────────────────────
# A tool is just a normal Python function the model is ALLOWED to call.
# The @tool decorator exposes it; the model reads the function's name,
# docstring, and type hints to decide WHEN and HOW to use it.

from langchain_core.tools import tool

@tool
def calcular_imc(weight_kg:float, height_m:float) -> str:
    """Calcula el índice de masa corporal (IMC) dado el peso (weight_kg) en kg y la altura (height_m) en metros."""
    imc = weight_kg / (height_m ** 2)

    if imc < 18.5:
        category = "Bajo peso"
    elif imc < 25:
        category = "Peso Normal"
    elif imc < 30:
        category = "Sobrepeso"
    else:
        category = "Obesidad"
    
    return f"IMC: {imc:.2f}, Categoría: {category}"

@tool
def obtener_total_palabras(texto:str) -> str:
    """Cuenta el número total de palabras en un texto dado."""
    total_palabras = len(texto.split())
    return f"Total de palabras: {total_palabras}"


# This is exactly what the model "sees" about a tool — the same info it
# uses to decide whether the tool fits the question.
print("=== Tool Info ===")
print(f"Name: {calcular_imc.name}")
print(f"Description: {calcular_imc.description}")
print(f"Args: {calcular_imc.args}")
print()

# bind_tools tells the model "these tools are available to you."
model_with_tools = model.bind_tools([calcular_imc, obtener_total_palabras])

# IMPORTANT (say this out loud): the model only *requests* a tool call —
# it does NOT run the tool. It hands back which tool to call and with what
# arguments. Actually executing the tool and looping the result back is the
# AGENT's job.
response = model_with_tools.invoke("Calcula el IMC para 87.80 kg y 1.72 m de altura")

print("=== Tool Call Response ===")
print(f"Tool calls: {response.tool_calls}")
print(f"Model output: {response.content}")
print()