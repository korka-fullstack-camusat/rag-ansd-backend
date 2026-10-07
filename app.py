import streamlit as st

from rag import answer_question

st.set_page_config(page_title="ANSD Chatbot", page_icon="📊")
st.title("📊 ANSD Chatbot")
st.caption("Assistant RAG sur les publications de l'ANSD Senegal")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bonjour ! Je suis votre assistant ANSD. Posez votre question pour commencer."}
    ]


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for h in sources:
            label = f"**[{h['source']}]({h['url']})**" if h.get("url") else f"**{h['source']}**"
            st.markdown(f"- {label} (p.{h['page']}, score {h['score']:.2f})")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

if question := st.chat_input("Votre message..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche dans les documents ANSD..."):
            try:
                result = answer_question(question)
            except Exception as exc:
                result = {"answer": f"Erreur : {exc}", "sources": []}
        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
