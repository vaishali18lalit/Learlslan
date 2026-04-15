"""
AI Advisor chat module for the Learslán dashboard.

Supports two modes:
  - popover_mode=True: uses text_input+button (works inside st.popover)
  - popover_mode=False: uses chat_input (full page mode)
"""

import logging
from textwrap import dedent

import streamlit as st
import pandas as pd

from insights.context import build_area_context, build_page_context
from insights.rag_engine import get_rag_engine

logger = logging.getLogger(__name__)


@st.cache_resource
def _get_cached_rag_engine():
    return get_rag_engine()

_MAX_HISTORY = 20
from config import GEMINI_API_KEY, GEMINI_MODEL

_SYSTEM_PROMPT = dedent("""\
    You are **Learslán AI**, an expert Irish community intelligence advisor built
    into the Learslán dashboard. You help users understand cost-of-living data,
    housing affordability, risk scores, transport connectivity, and energy
    efficiency across Ireland's 26 counties and 255 Electoral Divisions.

    PERSONALITY:
    - Warm, knowledgeable, concise. You speak like a trusted local advisor.
    - Use Irish place names naturally. Reference specific data points.
    - Be opinionated when the data supports it.

    CAPABILITIES:
    - Explain ML scores (risk, livability, transport, affordability) and what drives them.
    - Compare areas using real metrics.
    - Advise on relocation decisions based on user priorities.
    - Reference Irish housing policy, grants, and government schemes.
    - Interpret SHAP feature importance to explain why an area scores high/low.

    CITATION RULES (CRITICAL):
    - When referencing policy documents or reports, cite with [Source: document title]
    - When referencing data metrics, cite with [Data: metric_name = value]
    - When referencing SHAP/model analysis, cite with [Model: SHAP analysis]
    - Include at least one citation per answer when data or documents are relevant.
    - If you cannot ground an answer in the available data or documents, say so honestly.

    FORMATTING:
    - Use markdown for readability (bold, bullet points, headers).
    - Keep answers focused - 3-5 paragraphs max unless the user asks for detail.
    - Use EUR for currency, Irish conventions for place names.
""")


def render_advisor_tab(
    selected_county,
    scores_df,
    drivers=None,
    market=None,
    models=None,
    feature_names=None,
    ts_data=None,
    popover_mode=False,
):
    """Render the AI Advisor. popover_mode=True uses text_input for st.popover compatibility."""
    _ensure_session_state()

    st.session_state["_advisor_models"] = models or {}
    st.session_state["_advisor_feature_names"] = feature_names or []
    st.session_state["_advisor_scored_df"] = scores_df
    st.session_state["_advisor_ts_data"] = ts_data
    st.session_state["_advisor_county"] = selected_county

    area_context = build_area_context(selected_county, scores_df, drivers, market)
    page_ctx = build_page_context(active_tab="advisor", selected_county=selected_county)

    # Chatbot header
    if popover_mode:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px; padding-bottom:12px; border-bottom:1px solid rgba(128,128,128,0.2);">
            <div style="width:40px; height:40px; border-radius:50%; background:linear-gradient(135deg,#10b981,#3b82f6); display:flex; align-items:center; justify-content:center; font-size:20px; flex-shrink:0;">&#x1F916;</div>
            <div>
                <div style="font-weight:700; font-size:1rem;">Learsl\u00e1n AI Advisor</div>
                <div style="font-size:0.75rem; opacity:0.6;">&#x1F7E2; Online &middot; Powered by GBM + TOPSIS + SHAP + ARIMA</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### Learsl\u00e1n AI Advisor")
        st.caption(f"Context: **{selected_county}** | Powered by live ML models")

    # Welcome message if no history
    msgs = st.session_state.advisor_messages
    if not msgs:
        with st.chat_message("assistant", avatar="\U0001f916"):
            st.markdown(
                f"Hi! I'm your **Learsl\u00e1n AI Advisor**. I have live access to ML models "
                f"covering **255 Electoral Divisions** across Ireland.\n\n"
                f"Currently looking at **{selected_county}**. Try asking me:\n"
                f"- *Where should I live on a \u20ac65k salary?*\n"
                f"- *Why is Dublin's risk score high?*\n"
                f"- *What will rent be in 6 months?*\n"
                f"- *What grants are available for first-time buyers?*"
            )

    # Show chat history
    display_msgs = msgs[-6:] if popover_mode else msgs
    for msg in display_msgs:
        avatar = "\U0001f916" if msg["role"] == "assistant" else "\U0001f464"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Input handling
    prompt = None
    if popover_mode:
        col1, col2 = st.columns([8, 1])
        with col1:
            user_input = st.text_input(
                "Ask anything...",
                key="popover_advisor_input",
                label_visibility="collapsed",
                placeholder="Ask about rent, areas, grants...",
            )
        with col2:
            send_clicked = st.button(">", key="popover_send_btn", type="primary", use_container_width=True)
        if user_input and send_clicked:
            prompt = user_input
    else:
        prompt = st.chat_input("Ask Learslán AI about this area...", key="advisor_chat_input")

    if prompt:
        st.session_state.advisor_messages.append({"role": "user", "content": prompt})
        with st.spinner("Running ML models + generating response..."):
            response = _generate_response(prompt, area_context, page_ctx)
        st.session_state.advisor_messages.append({"role": "assistant", "content": response})
        _trim_history()
        st.rerun()


def render_floating_advisor(
    selected_county,
    scores_df,
    drivers=None,
    market=None,
    page_context=None,
):
    """Render the AI Advisor as a sidebar expander (visible on all tabs)."""
    _ensure_session_state()

    area_context = build_area_context(selected_county, scores_df, drivers, market)
    page_ctx = page_context or build_page_context(
        active_tab="overview", selected_county=selected_county,
    )

    with st.sidebar:
        st.markdown("---")
        with st.expander("AI Advisor", expanded=False):
            tab_label = page_ctx.get("active_tab", "overview").title()
            st.caption(f"Looking at: **{tab_label}** - **{selected_county}**")

            for msg in st.session_state.advisor_messages[-6:]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Ask Learslán AI...", key="floating_advisor_input"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.advisor_messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = _generate_response(prompt, area_context, page_ctx)
                    st.markdown(response)
                st.session_state.advisor_messages.append({"role": "assistant", "content": response})

                _trim_history()


# -- Internal --

def _ensure_session_state():
    if "advisor_messages" not in st.session_state:
        st.session_state.advisor_messages = []


def _trim_history():
    if len(st.session_state.advisor_messages) > _MAX_HISTORY:
        st.session_state.advisor_messages = st.session_state.advisor_messages[-_MAX_HISTORY:]


def _generate_response(query, area_context, page_context):
    from insights.ml_tools import build_ml_context
    ml_context = ""
    try:
        ml_context = build_ml_context(
            query=query,
            scored_df=st.session_state.get("_advisor_scored_df", pd.DataFrame()),
            models=st.session_state.get("_advisor_models", {}),
            feature_names=st.session_state.get("_advisor_feature_names", []),
            selected_county=st.session_state.get("_advisor_county", ""),
            ts_data=st.session_state.get("_advisor_ts_data"),
        )
    except Exception as e:
        logger.warning("ML inference failed: %s", e)

    rag_engine = _get_cached_rag_engine()
    rag_results = rag_engine.retrieve(query, top_k=3)

    full_system = _assemble_system_prompt(area_context, page_context, rag_results, ml_context)

    try:
        return _call_llm(query, full_system)
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return _template_fallback(query, area_context, rag_results, error=str(e))


def _assemble_system_prompt(area_context, page_context, rag_results, ml_context=""):
    parts = [_SYSTEM_PROMPT]

    desc = page_context.get("natural_description", "browsing the dashboard")
    parts.append(f"\n--- CURRENT PAGE CONTEXT ---\nThe user is currently {desc}.")
    parts.append(f"\n--- AREA DATA (from pre-trained GBM models) ---\n{area_context}")

    if ml_context:
        parts.append(f"\n--- LIVE MODEL INFERENCE (run just now for this query) ---{ml_context}")
        parts.append(
            "\nIMPORTANT: The live model results above were computed in real-time "
            "using the trained ML models. Prioritize these over static data when answering. "
            "Always cite the model used, e.g. [Model: TOPSIS ranking] or [Model: SHAP analysis]."
        )

    if rag_results:
        parts.append("\n--- REFERENCE DOCUMENTS ---")
        for i, r in enumerate(rag_results, 1):
            source = r["source_title"]
            authority = r["authority"]
            method = r.get("method", "unknown")
            parts.append(
                f"[{i}] {source}"
                + (f" (Authority: {authority})" if authority else "")
                + f" [retrieval: {method}, score: {r['score']:.3f}]"
                + f"\n    {r['content'][:400]}"
            )
        parts.append(
            "\nUse these references to ground your answers. "
            "Cite them as [Source: document title] when relevant."
        )

    return "\n".join(parts)


def _call_llm(query, system_prompt):
    from google import genai
    from google.genai import types

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not configured")

    client = genai.Client(api_key=GEMINI_API_KEY)

    contents = []
    for msg in st.session_state.get("advisor_messages", [])[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=query)]))

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=1024,
            temperature=0.7,
        ),
    )
    return response.text


def _template_fallback(query, area_context, rag_results, error=None):
    lines = area_context.split("\n")
    county = lines[0].replace("=== ", "").replace(" -- Key Metrics ===", "") if lines else "this area"

    metrics = {}
    for line in lines:
        if ":" in line and "===" not in line and "---" not in line:
            key, val = line.split(":", 1)
            metrics[key.strip()] = val.strip()

    response_parts = [f"## {county} -- Quick Summary\n"]
    risk = metrics.get("Risk Score", "N/A")
    livability = metrics.get("Livability Score", "N/A")
    affordability = metrics.get("Affordability Score", "N/A")
    rent = metrics.get("Avg Monthly Rent", "N/A")

    response_parts.append(
        f"**{county}** has a risk score of **{risk}**, "
        f"livability of **{livability}**, and affordability of **{affordability}**. "
        f"Average monthly rent is **{rent}**.\n"
    )

    if rag_results:
        response_parts.append("### Relevant Policy Context\n")
        for r in rag_results[:2]:
            response_parts.append(
                f"> {r['content'][:200]}...\n"
                f"> -- [Source: {r['source_title']}]\n"
            )

    response_parts.append(
        "\n*AI Advisor is running in template mode. "
        "Configure GEMINI_API_KEY in .env for full conversational AI.*"
    )

    if error:
        response_parts.append(f"\n\n`Debug: {error}`")

    return "\n".join(response_parts)
