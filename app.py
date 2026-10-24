"""Servidor web do chatbot.

Uso:
    uvicorn app:app --reload
    # abra http://localhost:8000
"""

import logging
from pathlib import Path

import anthropic
import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chatbot import carregar_negocio, provedor, responder

PASTA_STATIC = Path(__file__).parent / "static"
log = logging.getLogger("uvicorn.error")

app = FastAPI(title="Chatbot de atendimento")
negocio = carregar_negocio()
log.info("Chatbot usando: %s", provedor())  # falha já na subida se a configuração estiver errada


class Mensagem(BaseModel):
    role: str
    content: str


class Conversa(BaseModel):
    mensagens: list[Mensagem]


@app.get("/")
def pagina():
    return FileResponse(PASTA_STATIC / "index.html")


@app.get("/api/status")
def status():
    nome = provedor()
    return {"negocio": negocio["nome"], "modo": "demonstracao" if nome == "demonstracao" else "ia", "provedor": nome}


@app.post("/api/chat")
def chat(conversa: Conversa):
    try:
        texto = responder(negocio, [m.model_dump() for m in conversa.mensagens])
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    except (anthropic.RateLimitError, openai.RateLimitError) as erro:
        log.warning("Limite de uso da IA: %s", erro)
        raise HTTPException(status_code=503, detail="Muitas mensagens agora. Tente de novo em instantes.")
    except (anthropic.APIError, openai.APIError) as erro:
        log.error("Erro da IA: %s", erro)
        raise HTTPException(status_code=502, detail="O atendente está indisponível no momento.")
    return {"resposta": texto}