"""
chatbot.py  —  AI Medical Assistant powered by Claude API
Place at ROOT of your project (same level as app.py)

INSTALL:
    pip install anthropic

SETUP:
    Create a .env file at project root:
        ANTHROPIC_API_KEY=sk-ant-...

    OR set it in your terminal:
        set ANTHROPIC_API_KEY=sk-ant-...      (Windows)
        export ANTHROPIC_API_KEY=sk-ant-...   (Mac/Linux)

    Get your API key at: https://console.anthropic.com
"""

import os
import streamlit as st
import anthropic
from typing import Optional

# ── Load API key ──────────────────────────────────────────────────
def _get_api_key() -> Optional[str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.environ.get("ANTHROPIC_API_KEY")
        except ImportError:
            pass
    return key


# ── System prompt ─────────────────────────────────────────────────
def _build_system_prompt(prediction_context: Optional[dict] = None) -> str:
    base = """You are a medical AI assistant specialized in cervical cancer risk analysis.
You help users understand their risk prediction results, explain the factors that influence risk,
and provide general health education about cervical cancer prevention.

IMPORTANT RULES:
- Always remind users you are an AI assistant, not a doctor
- Never give a definitive medical diagnosis
- Always recommend consulting a qualified healthcare professional for medical decisions
- Be compassionate, clear, and professional
- Keep responses concise (3-5 sentences max unless more detail is requested)
- Use simple language — avoid complex medical jargon
- If asked about something unrelated to cervical cancer or health, politely redirect

ABOUT THE PROJECT:
- This dashboard predicts cervical cancer risk using a machine learning model
- Dataset: UCI Cervical Cancer Risk Factors (858 patients, 36 features)
- The model uses: age, sexual history, smoking, contraceptives, STDs, etc.
- Key metric is Recall (minimizing missed cancers is critical)
"""
    if prediction_context:
        prob   = prediction_context.get("probability", 0)
        level  = prediction_context.get("risk_level", "unknown")
        factors = prediction_context.get("risk_factors", [])
        base += f"""
CURRENT PATIENT CONTEXT (use this to personalize your answers):
- Risk level: {level}
- Probability score: {prob:.1%}
- Identified risk factors: {', '.join(factors) if factors else 'none detected'}

You can reference this patient's specific results when answering questions.
"""
    return base


# ── Main chatbot function (call this from app.py) ─────────────────
def render_chatbot(prediction_context: Optional[dict] = None):
    """
    Renders the full chatbot UI inside a Streamlit tab or section.

    Args:
        prediction_context: dict with keys 'probability', 'risk_level', 'risk_factors'
                            Pass this after a prediction to give the AI patient context.
    Example:
        from chatbot import render_chatbot
        render_chatbot(prediction_context={
            "probability": 0.72,
            "risk_level": "HIGH",
            "risk_factors": ["Age > 40", "Smoking 12 years", "STDs (3)"]
        })
    """

    # ── Styles ────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .chat-bubble-user {
        background: #1e2530;
        border: 1px solid #30363d;
        border-radius: 12px 12px 4px 12px;
        padding: 10px 14px;
        margin: 6px 0 6px 60px;
        font-size: 0.88rem;
        color: #e6edf3;
        line-height: 1.55;
    }
    .chat-bubble-ai {
        background: #0c1b2e;
        border: 1px solid #185FA5;
        border-left: 3px solid #58a6ff;
        border-radius: 4px 12px 12px 12px;
        padding: 10px 14px;
        margin: 6px 60px 6px 0;
        font-size: 0.88rem;
        color: #79c0ff;
        line-height: 1.55;
    }
    .chat-bubble-error {
        background: #1c0a0a;
        border-left: 3px solid #da3633;
        border-radius: 4px 12px 12px 12px;
        padding: 10px 14px;
        margin: 6px 60px 6px 0;
        font-size: 0.88rem;
        color: #ff7b72;
    }
    .chat-label-user { font-size:0.72rem; color:#8b949e; text-align:right;  margin: 0 0 2px; }
    .chat-label-ai   { font-size:0.72rem; color:#58a6ff; text-align:left;   margin: 0 0 2px; }
    .chat-header {
        display: flex; align-items: center; gap: 10px;
        padding: 10px 14px;
        background: #0d1117;
        border: 1px solid #1e2530;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────
    api_key = _get_api_key()
    status_color = "#3fb950" if api_key else "#da3633"
    status_text  = "Connected · Claude claude-sonnet-4-5" if api_key else "API key missing"

    st.markdown(f"""
    <div class='chat-header'>
        <div style='width:8px;height:8px;border-radius:50%;background:{status_color};flex-shrink:0'></div>
        <span style='font-size:0.85rem;font-weight:500;color:#e6edf3'>AI Medical Assistant</span>
        <span style='font-size:0.75rem;color:#8b949e;margin-left:auto'>{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    if not api_key:
        st.markdown("""
        <div style='background:#1c1600;border-left:3px solid #d29922;border-radius:6px;
                    padding:12px 16px;font-size:0.84rem;color:#e3b341;margin-bottom:12px'>
            <b>Setup required:</b><br>
            1. Get your free API key at <b>console.anthropic.com</b><br>
            2. Create a <code>.env</code> file in your project root<br>
            3. Add: <code>ANTHROPIC_API_KEY=sk-ant-your-key-here</code><br>
            4. Install: <code>pip install anthropic python-dotenv</code><br>
            5. Restart Streamlit
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Context badge ─────────────────────────────────────────────
    if prediction_context:
        prob  = prediction_context.get("probability", 0)
        level = prediction_context.get("risk_level", "?")
        color = "#da3633" if level == "HIGH" else "#238636"
        st.markdown(f"""
        <div style='background:#0d1117;border:1px solid {color};border-radius:8px;
                    padding:8px 14px;font-size:0.78rem;color:{color};margin-bottom:12px;
                    font-family:IBM Plex Mono,monospace'>
            Patient context loaded — Risk: {level} ({prob:.1%}) — Ask me to explain
        </div>
        """, unsafe_allow_html=True)

    # ── Suggested questions ───────────────────────────────────────
    if "chat_history" not in st.session_state or len(st.session_state.chat_history) == 0:
        st.markdown("<p style='font-size:0.78rem;color:#8b949e;margin-bottom:6px'>Suggested questions:</p>",
                    unsafe_allow_html=True)
        suggestions = [
            "Why am I at risk?",
            "What does Recall mean in ML?",
            "How can I reduce my risk?",
            "What is SHAP and how does it work?",
            "What are the main risk factors for cervical cancer?",
        ]
        cols = st.columns(len(suggestions))
        for col, q in zip(cols, suggestions):
            if col.button(q, key=f"sug_{q[:10]}", width="stretch"):
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.session_state.chat_pending = True
                st.rerun()

    # ── Init chat history ─────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_pending = False

    # ── Render existing messages ──────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<p class='chat-label-user'>You</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='chat-label-ai'>AI Assistant</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-bubble-ai'>{msg['content']}</div>",
                            unsafe_allow_html=True)

    # ── Process pending message ───────────────────────────────────
    if st.session_state.get("chat_pending") and st.session_state.chat_history:
        last_user_msg = st.session_state.chat_history[-1]["content"]
        with st.spinner("AI is thinking..."):
            response = _call_claude(
                messages=st.session_state.chat_history,
                system_prompt=_build_system_prompt(prediction_context),
                api_key=api_key
            )
        if response:
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Sorry, I encountered an error. Please check your API key and try again."
            })
        st.session_state.chat_pending = False
        st.rerun()

    # ── Input area ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_input, col_send, col_clear = st.columns([6, 1, 1])

    with col_input:
        user_input = st.text_input(
            "Ask the AI assistant",
            placeholder="e.g. Why is my risk high? What does Recall mean?",
            label_visibility="collapsed",
            key="chat_input"
        )

    with col_send:
        send = st.button("Send", width="stretch", type="primary")

    with col_clear:
        if st.button("Clear", width="stretch"):
            st.session_state.chat_history = []
            st.session_state.chat_pending = False
            st.rerun()

    if (send or user_input) and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.session_state.chat_pending = True
        st.rerun()

    # ── Disclaimer ────────────────────────────────────────────────
    st.markdown("""
    <p style='font-size:0.72rem;color:#30363d;margin-top:8px;text-align:center'>
        AI assistant for educational purposes only · Not a substitute for medical advice
    </p>
    """, unsafe_allow_html=True)


# ── Claude API call ───────────────────────────────────────────────
def _call_claude(messages: list, system_prompt: str, api_key: str) -> Optional[str]:
    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Convert history to Claude format (max last 10 messages to save tokens)
        history = messages[-10:]
        claude_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in history
            if m["role"] in ("user", "assistant")
        ]

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            system=system_prompt,
            messages=claude_messages
        )
        return response.content[0].text

    except anthropic.AuthenticationError:
        return "Authentication failed. Please check your ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        return "Rate limit reached. Please wait a moment and try again."
    except Exception as e:
        return f"Error: {str(e)}"
