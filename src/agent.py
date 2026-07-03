from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

try:
    from agno.agent import Agent
except Exception:  # pragma: no cover - import varies by installed Agno version
    try:
        from agno import Agent  # type: ignore
    except Exception:  # pragma: no cover
        Agent = None  # type: ignore


@dataclass
class AuditAgent:
    provider: str
    model_name: str
    agent: Any

    def answer_question(self, question: str, context: str, history: list[dict[str, str]]) -> str:
        prompt = build_chat_prompt(question=question, context=context, history=history)
        return _run_agent(self.agent, prompt)


def _ensure_api_environment(provider: str) -> None:
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        return

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            os.environ["GOOGLE_API_KEY"] = api_key


def _build_model(provider: str, model_name: str) -> Any:
    if provider == "openai":
        for import_path in (
            ("agno.models.openai", "OpenAIChat"),
            ("agno.models.openai.chat", "OpenAIChat"),
        ):
            module_name, class_name = import_path
            try:
                module = __import__(module_name, fromlist=[class_name])
                return getattr(module, class_name)(id=model_name)
            except Exception:
                continue
        raise ImportError("Nao foi possivel importar o modelo OpenAI do Agno.")

    if provider == "gemini":
        for import_path in (
            ("agno.models.google", "Gemini"),
            ("agno.models.google_genai", "Gemini"),
        ):
            module_name, class_name = import_path
            try:
                module = __import__(module_name, fromlist=[class_name])
                return getattr(module, class_name)(id=model_name)
            except Exception:
                continue
        raise ImportError("Nao foi possivel importar o modelo Gemini do Agno.")

    raise ValueError("Provider invalido. Use openai ou gemini.")


def _run_agent(agent: Any, prompt: str) -> str:
    if agent is None:
        return (
            "Agno nao esta disponivel neste ambiente. "
            "Instale a dependencia `agno` para habilitar o chat e o relatorio de IA."
        )

    response = agent.run(prompt)
    if hasattr(response, "content"):
        return str(response.content).strip()
    return str(response).strip()


def build_audit_agent(provider: str, model_name: str) -> AuditAgent:
    if Agent is None:
        raise ImportError(
            "Nao foi possivel importar Agno. Verifique se a dependencia agno esta instalada."
        )

    provider = provider.strip().lower()
    model_name = model_name.strip()
    if not model_name:
        raise ValueError("Informe um nome de modelo valido na barra lateral.")

    _ensure_api_environment(provider)
    model = _build_model(provider, model_name)

    instructions = [
        "Você é um auditor BIM especialista em IFC.",
        "Trabalhe apenas com o contexto fornecido pelo sistema e com o resumo estruturado do modelo.",
        "Nao invente dados que nao estejam presentes no contexto.",
        "Explique de forma clara, tecnica e objetiva.",
        "Responda em portugues do Brasil.",
    ]

    try:
        agent = Agent(
            model=model,
            instructions=instructions,
            markdown=True,
        )
    except TypeError:
        agent = Agent(
            model=model,
            system_message="\n".join(instructions),
            markdown=True,
        )
    return AuditAgent(provider=provider, model_name=model_name, agent=agent)


def build_audit_summary_prompt(analysis: Any) -> str:
    payload = analysis.to_serializable() if hasattr(analysis, "to_serializable") else analysis
    return f"""
Voce e um auditor BIM especializado em IFC.

Analise o resumo estruturado abaixo e produza:
1. resumo tecnico do modelo;
2. pontos criticos;
3. nivel de completude informacional;
4. recomendacoes de melhoria;
5. conclusao de auditoria.

Regras:
- baseie a resposta apenas nos dados fornecidos;
- nao cite informacoes que nao estejam no resumo;
- escreva em portugues do Brasil;
- use formato Markdown com titulos curtos.

Resumo estruturado do IFC:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def generate_audit_report(agent_bundle: AuditAgent, analysis: Any) -> str:
    prompt = build_audit_summary_prompt(analysis)
    return _run_agent(agent_bundle.agent, prompt)


def build_chat_context(analysis: Any, report: str | None) -> str:
    payload = analysis.to_serializable() if hasattr(analysis, "to_serializable") else analysis
    return json.dumps(
        {
            "ifc_summary": payload,
            "previous_report": report or "",
        },
        ensure_ascii=False,
        indent=2,
    )


def build_chat_prompt(question: str, context: str, history: list[dict[str, str]]) -> str:
    conversation = "\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in history[-10:]
    )
    return f"""
Voce e um assistente tecnico de auditoria BIM especializado no IFC analisado.

Use apenas o contexto estruturado abaixo:
{context}

Conversa recente:
{conversation if conversation else 'Sem historico anterior.'}

Pergunta do usuario:
{question}

Responda em portugues do Brasil, com objetividade e sem inventar dados.
""".strip()
