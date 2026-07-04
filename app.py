from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from src.agent import build_audit_agent, build_chat_context, generate_audit_report
from src.ifc_analyzer import analyze_ifc_file, format_analysis_for_display


st.set_page_config(
    page_title="IFC Audit Agent",
    page_icon="🏗️",
    layout="wide",
)


MODEL_OPTIONS = {
    "gemini": ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
}

MODEL_DEFAULTS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}


def ensure_session_state() -> None:
    st.session_state.setdefault("ifc_analysis", None)
    st.session_state.setdefault("ifc_agent_report", None)
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("last_uploaded_name", None)
    st.session_state.setdefault("selected_llm_provider", None)
    st.session_state.setdefault("selected_llm_model", None)


def reset_chat() -> None:
    st.session_state.chat_messages = []


def _secret_lookup(secret_key: str) -> str:
    try:
        value = st.secrets.get(secret_key, "")
        if isinstance(value, str):
            return value.strip()
    except Exception:
        pass

    try:
        grouped = st.secrets.get("api_keys", {})
        if isinstance(grouped, dict):
            value = grouped.get(secret_key, "")
            if isinstance(value, str):
                return value.strip()
    except Exception:
        pass

    return ""


def _session_key_lookup(provider: str) -> str:
    provider = provider.strip().lower()
    key_name = "api_key_gemini" if provider == "gemini" else "api_key_openai"
    value = st.session_state.get(key_name, "")
    if isinstance(value, str):
        return value.strip()
    return ""


def _provider_key_available(provider: str) -> bool:
    provider = provider.strip().lower()
    if provider == "openai":
        return bool(
            _session_key_lookup("openai")
            or os.getenv("OPENAI_API_KEY", "").strip()
            or _secret_lookup("OPENAI_API_KEY")
        )

    if provider == "gemini":
        return bool(
            _session_key_lookup("gemini")
            or os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GOOGLE_API_KEY", "").strip()
            or _secret_lookup("GEMINI_API_KEY")
            or _secret_lookup("GOOGLE_API_KEY")
        )

    return False


def _suggest_provider(env_provider: str) -> str:
    env_provider = env_provider.strip().lower()
    gemini_available = _provider_key_available("gemini")
    openai_available = _provider_key_available("openai")

    if gemini_available and not openai_available:
        return "gemini"
    if openai_available and not gemini_available:
        return "openai"
    if env_provider in {"gemini", "openai"}:
        return env_provider
    if gemini_available and openai_available:
        return "gemini"
    return "gemini"


def resolve_api_key(provider: str, typed_key: str) -> tuple[str, str]:
    provider = provider.strip().lower()
    typed_key = typed_key.strip()
    if typed_key:
        return typed_key, "interface"

    if provider == "openai":
        env_key = os.getenv("OPENAI_API_KEY", "").strip()
        if env_key:
            return env_key, "ambiente"
        secret_key = _secret_lookup("OPENAI_API_KEY")
        if secret_key:
            return secret_key, "st.secrets"
        return "", ""

    if provider == "gemini":
        env_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
        if env_key:
            return env_key, "ambiente"
        secret_key = _secret_lookup("GEMINI_API_KEY")
        if secret_key:
            return secret_key, "st.secrets"
        secret_key = _secret_lookup("GOOGLE_API_KEY")
        if secret_key:
            return secret_key, "st.secrets"
        return "", ""

    return "", ""


def format_llm_error(provider: str, exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()

    if any(
        token in lowered
        for token in (
            "incorrect api key",
            "invalid api key",
            "api key provided",
            "authentication",
            "unauthorized",
            "unauthenticated",
        )
    ):
        if provider == "gemini":
            return (
                "A chave do Gemini foi recusada. Verifique se o valor informado em GEMINI_API_KEY "
                "está correto e corresponde ao provider selecionado."
            )
        return (
            "A chave da OpenAI foi recusada. Verifique se o valor informado em OPENAI_API_KEY "
            "está correto e corresponde ao provider selecionado."
        )

    return f"Falha ao gerar resposta com IA: {exc}"


def render_dataframe(title: str, frame: pd.DataFrame) -> None:
    st.subheader(title)
    if frame.empty:
        st.info("Nenhum registro encontrado.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def main() -> None:
    ensure_session_state()

    st.title("IFC Audit Agent")
    st.caption(
        "Aplicação Streamlit para auditar a qualidade informacional de um modelo BIM IFC com ifcopenshell + Agno."
    )

    with st.sidebar:
        st.header("Configuração do LLM")
        env_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

        if st.session_state.selected_llm_provider not in {"openai", "gemini"}:
            st.session_state.selected_llm_provider = _suggest_provider(env_provider)

        provider = st.selectbox(
            "Provider",
            options=["openai", "gemini"],
            key="selected_llm_provider",
        )

        model_options = MODEL_OPTIONS[provider]
        current_model = st.session_state.get("selected_llm_model")
        if current_model not in model_options:
            current_model = MODEL_DEFAULTS[provider]
            st.session_state.selected_llm_model = current_model

        model_index = model_options.index(current_model)
        model_name = st.selectbox(
            "Modelo",
            options=model_options,
            index=model_index,
            key=f"model_widget_{provider}",
        )
        st.session_state.selected_llm_model = model_name

        api_label = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        typed_api_key = st.text_input(
            api_label,
            type="password",
            key=f"api_key_{provider}",
            placeholder=f"Digite {api_label} aqui",
            help=(
                "A chave digitada aqui tem prioridade sobre variáveis de ambiente e st.secrets."
            ),
        )
        api_key, api_source = resolve_api_key(provider, typed_api_key)

        if typed_api_key:
            st.caption("A chave informada na interface será usada nesta sessão.")
        elif api_key:
            st.caption(f"Usando chave carregada de {api_source}.")
        else:
            st.warning(
                f"Nenhuma chave encontrada para {api_label}. Digite a chave ou configure o ambiente/st.secrets."
            )

        st.caption(f"Provider ativo: {provider}")
        st.caption(f"Modelo compatível: {model_name}")

        st.divider()
        st.subheader("Arquivo IFC")
        uploaded_ifc = st.file_uploader(
            "Envie um arquivo .ifc",
            type=["ifc"],
            accept_multiple_files=False,
        )

    if uploaded_ifc is None:
        st.info("Envie um arquivo IFC para iniciar a auditoria.")
        return

    if st.session_state.last_uploaded_name != uploaded_ifc.name:
        st.session_state.ifc_analysis = None
        st.session_state.ifc_agent_report = None
        reset_chat()
        st.session_state.last_uploaded_name = uploaded_ifc.name

    if st.session_state.ifc_analysis is None:
        with st.spinner("Lendo o modelo IFC e extraindo dados..."):
            try:
                st.session_state.ifc_analysis = analyze_ifc_file(
                    uploaded_ifc.getvalue(),
                    file_name=uploaded_ifc.name,
                )
            except Exception as exc:
                st.error(f"Não foi possível carregar o IFC: {exc}")
                st.stop()

    analysis = st.session_state.ifc_analysis
    display = format_analysis_for_display(analysis)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de entidades", display["total_entities"])
    col2.metric("Score de completude", f"{display['completeness_score']:.1f}%")
    col3.metric("Projeto", display["project_name"] or "Não informado")

    tab_overview, tab_tables, tab_report, tab_chat = st.tabs(
        ["Visão geral", "Tabelas", "Relatório IA", "Chat"]
    )

    with tab_overview:
        st.subheader("Resumo técnico")
        st.write(
            "O agente faz uma auditoria estrutural do IFC, sem análise geométrica complexa."
        )

        overview_left, overview_right = st.columns(2)
        with overview_left:
            st.write("**Nome do projeto**")
            st.write(display["project_name"] or "Não encontrado")
            st.write("**Quantidade total de entidades**")
            st.write(display["total_entities"])
            st.write("**Entidades sem Name**")
            st.write(display["missing_name_count"])
            st.write("**Entidades sem material**")
            st.write(display["missing_material_count"])

        with overview_right:
            st.write("**Entidades sem propriedades IFC**")
            st.write(display["missing_pset_count"])
            st.write("**Entidades sem quantidades**")
            st.write(display["missing_quantity_count"])
            st.write("**Nível de completude informacional**")
            st.write(display["completeness_label"])
            st.write("**Conclusão preliminar**")
            st.write(display["preliminary_conclusion"])

    with tab_tables:
        render_dataframe("Contagem por classe IFC", display["class_counts"])
        render_dataframe("Pavimentos IfcBuildingStorey", display["storeys"])
        render_dataframe("Elementos sem Name", display["missing_name"])
        render_dataframe("Elementos sem material", display["missing_material"])
        render_dataframe("Elementos sem propriedades IFC", display["missing_psets"])
        render_dataframe("Elementos sem quantidades", display["missing_quantities"])
        render_dataframe("Resumo dos principais tipos", display["main_types"])

    with tab_report:
        st.subheader("Relatório do agente")

        if st.button("Gerar análise técnica com Agno", type="primary"):
            try:
                if not api_key:
                    st.error(
                        "Informe a chave de API na sidebar ou configure o ambiente/st.secrets antes de gerar o relatório."
                    )
                    st.stop()
                with st.spinner("O agente está analisando o resumo estruturado do IFC..."):
                    agent = build_audit_agent(
                        provider=provider,
                        model_name=model_name,
                        api_key=api_key,
                    )
                    report = generate_audit_report(agent, analysis)
                    st.session_state.ifc_agent_report = report
            except Exception as exc:
                st.error(format_llm_error(provider, exc))

        if st.session_state.ifc_agent_report:
            st.markdown(st.session_state.ifc_agent_report)
        else:
            st.info("Clique no botão para gerar o relatório técnico automatizado.")

    with tab_chat:
        st.subheader("Chat sobre o IFC")
        st.caption(
            "Faça perguntas sobre o modelo analisado. O agente responde com base no resumo extraído do IFC."
        )

        chat_context = build_chat_context(analysis, st.session_state.ifc_agent_report)

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_message = st.chat_input("Pergunte algo sobre o IFC analisado")
        if user_message:
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_message}
            )
            with st.chat_message("user"):
                st.markdown(user_message)

            try:
                if not api_key:
                    raise ValueError(
                        "Informe a chave de API na sidebar ou configure o ambiente/st.secrets."
                    )
                agent = build_audit_agent(
                    provider=provider,
                    model_name=model_name,
                    api_key=api_key,
                )
                answer = agent.answer_question(
                    question=user_message,
                    context=chat_context,
                    history=st.session_state.chat_messages,
                )
            except Exception as exc:
                answer = format_llm_error(provider, exc)

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer}
            )
            with st.chat_message("assistant"):
                st.markdown(answer)

        if st.button("Limpar chat"):
            reset_chat()
            st.rerun()

    with st.expander("Resumo estruturado enviado ao agente"):
        st.code(json.dumps(analysis.to_serializable(), ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
