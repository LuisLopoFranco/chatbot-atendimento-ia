"""Lógica do atendente virtual: monta o contexto do negócio e responde ao cliente.

"""

import json
import os
import unicodedata
from pathlib import Path

import anthropic
import openai

MODELO_CLAUDE = "claude-opus-5-5"
# Fornecedores que seguem o formato de API da OpenAI: endereço, variável da chave e modelo padrão.
COMPATIVEIS_OPENAI = {
    "ollama": {"url": "http://localhost:11434/v1", "chave": None, "modelo": "llama3.2"},
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "chave": "GEMINI_API_KEY",
        "modelo": "gemini-flash-latest",
    },
    "groq": {"url": "https://api.groq.com/openai/v1", "chave": "GROQ_API_KEY", "modelo": "llama-3.3-70b-versatile"},
}
MAX_MENSAGENS = 20
MAX_CARACTERES = 2000
ARQUIVO_NEGOCIO = Path(__file__).parent / "negocio.json"


def carregar_negocio(caminho: Path = ARQUIVO_NEGOCIO) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def montar_instrucoes(negocio: dict) -> str:
    return f"""Você é o atendente virtual da {negocio['nome']}, conversando com clientes pelo site ou WhatsApp.

Responda em português do Brasil, de forma simpática e curta (no máximo 4 frases ou uma lista breve).
Use somente as informações do negócio abaixo. Se o cliente perguntar algo que não está aqui,
diga que não tem essa informação e indique o WhatsApp {negocio['whatsapp']}.
Nunca invente preços, sabores, prazos ou promoções.

Quando o cliente quiser fazer um pedido, ajude a montar: sabores, tamanhos, bebidas, endereço
(confira se o bairro é atendido) e forma de pagamento. No fim, mostre um resumo com o valor total
(itens + taxa de entrega, quando houver) e diga que o pedido é confirmado pelo WhatsApp.

<negocio>
{json.dumps(negocio, ensure_ascii=False, indent=2)}
</negocio>"""


def validar_historico(mensagens: list[dict]) -> list[dict]:
    """Mantém só o formato esperado pela API e limita o tamanho da conversa."""
    limpas = []
    for m in mensagens[-MAX_MENSAGENS:]:
        papel = m.get("role")
        texto = str(m.get("content", "")).strip()[:MAX_CARACTERES]
        if papel in ("user", "assistant") and texto:
            limpas.append({"role": papel, "content": texto})
    while limpas and limpas[0]["role"] != "user":
        limpas.pop(0)
    if not limpas or limpas[-1]["role"] != "user":
        raise ValueError("A última mensagem precisa ser do cliente.")
    return limpas


def provedor() -> str:
    """Qual IA responde: claude, ollama, gemini, groq ou demonstracao."""
    escolhido = os.getenv("CHATBOT_PROVEDOR", "").strip().lower()
    if not escolhido:
        return "claude" if os.getenv("ANTHROPIC_API_KEY") else "demonstracao"
    if escolhido == "claude":
        variavel = "ANTHROPIC_API_KEY"
    elif escolhido in COMPATIVEIS_OPENAI:
        variavel = COMPATIVEIS_OPENAI[escolhido]["chave"]
    else:
        opcoes = ", ".join(["claude", *COMPATIVEIS_OPENAI])
        raise ValueError(f"CHATBOT_PROVEDOR inválido: {escolhido}. Use um destes: {opcoes}.")
    if variavel and not os.getenv(variavel):
        raise ValueError(f"Para usar {escolhido}, defina a variável {variavel} com a sua chave.")
    return escolhido


def modelo(nome_provedor: str) -> str:
    padrao = MODELO_CLAUDE if nome_provedor == "claude" else COMPATIVEIS_OPENAI[nome_provedor]["modelo"]
    return os.getenv("CHATBOT_MODELO", padrao)


def responder_com_claude(negocio: dict, mensagens: list[dict], cliente: anthropic.Anthropic | None = None) -> str:
    cliente = cliente or anthropic.Anthropic()
    resposta = cliente.beta.messages.create(
        model=modelo("claude"),
        max_tokens=8000,
        system=montar_instrucoes(negocio),
        messages=mensagens,
        output_config={"effort": "low"},
        cache_control={"type": "ephemeral"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resposta.stop_reason == "refusal":
        return f"Desculpe, não consigo ajudar com isso por aqui. Fale com a gente no WhatsApp {negocio['whatsapp']}."
    return "".join(bloco.text for bloco in resposta.content if bloco.type == "text").strip()


def responder_compativel_openai(
    negocio: dict, mensagens: list[dict], nome_provedor: str, cliente: openai.OpenAI | None = None
) -> str:
    """Responde com Ollama, Gemini, Groq ou outro serviço no formato da OpenAI."""
    if cliente is None:
        config = COMPATIVEIS_OPENAI[nome_provedor]
        # O Ollama não usa chave, mas o SDK da OpenAI exige uma.
        chave = os.getenv(config["chave"]) if config["chave"] else "ollama"
        cliente = openai.OpenAI(base_url=config["url"], api_key=chave)
    resposta = cliente.chat.completions.create(
        model=modelo(nome_provedor),
        messages=[{"role": "system", "content": montar_instrucoes(negocio)}, *mensagens],
    )
    texto = (resposta.choices[0].message.content or "").strip()
    return texto or f"Desculpe, não consegui responder agora. Fale com a gente no WhatsApp {negocio['whatsapp']}."


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.lower()


def responder_demo(negocio: dict, mensagens: list[dict]) -> str:
    """Respostas fixas por palavra-chave, para testar o site sem chave de API."""
    pergunta = _normalizar(mensagens[-1]["content"])

    if any(p in pergunta for p in ("horario", "abre", "fecha", "funciona")):
        linhas = [f"• {dia}: {hora}" for dia, hora in negocio["horarios"].items()]
        return "Nosso horário:\n" + "\n".join(linhas)
    if any(p in pergunta for p in ("entrega", "bairro", "taxa", "delivery", "demora")):
        e = negocio["entrega"]
        return (
            f"Entregamos em {', '.join(e['bairros'])}. Taxa de {e['taxa']}, "
            f"e o tempo médio é de {e['tempo_medio']}."
        )
    if any(p in pergunta for p in ("pag", "pix", "cartao", "dinheiro")):
        return "Aceitamos " + ", ".join(negocio["pagamento"]) + "."
    if any(p in pergunta for p in ("cardapio", "sabor", "preco", "quanto", "pizza", "menu")):
        linhas = [
            f"• {i['item']}: R$ {i['grande']:.2f} (grande)".replace(".", ",")
            for i in negocio["cardapio"]
            if "grande" in i
        ]
        return "Nossos sabores:\n" + "\n".join(linhas)
    if any(p in pergunta for p in ("onde", "endereco", "fica", "retirada")):
        return f"Estamos na {negocio['endereco']}. Você pode retirar no balcão!"
    return (
        "Estou em modo demonstração e só respondo sobre horário, entrega, pagamento, "
        f"cardápio e endereço. Para outras dúvidas, chame no WhatsApp {negocio['whatsapp']}."
    )


def responder(negocio: dict, mensagens: list[dict]) -> str:
    mensagens = validar_historico(mensagens)
    nome = provedor()
    if nome == "claude":
        return responder_com_claude(negocio, mensagens)
    if nome in COMPATIVEIS_OPENAI:
        return responder_compativel_openai(negocio, mensagens, nome)
    return responder_demo(negocio, mensagens)