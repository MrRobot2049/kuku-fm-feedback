"""Streamlit UI for evidence-based Kuku FM review triage."""

from pathlib import Path

import pandas as pd
import streamlit as st

import agentic_rag as rag


st.set_page_config(page_title="Kuku FM Review Triage", page_icon="🎧", layout="wide")


@st.cache_resource(show_spinner="Loading the review search models on first run…")
def get_retriever():
    return rag.DenseRetriever()


@st.cache_resource(show_spinner="Loading the relevance ranker on first run…")
def get_reranker():
    return rag.RelevanceReranker()


@st.cache_data
def get_corpus_stats():
    frame = pd.read_csv(rag.DATA_PATH)
    return len(frame), frame["at"].min(), frame["at"].max()


st.title("Kuku FM listener feedback")
st.write(
    "Ask a question about the public review sample. The app finds matching reviews "
    "and shows the evidence behind its answer."
)

count, first_date, last_date = get_corpus_stats()
st.caption(
    f"Sample: {count} English Google Play reviews from India · "
    f"{str(first_date)[:10]} to {str(last_date)[:10]} · reviewer names omitted"
)

with st.sidebar:
    st.header("Answer mode")
    mode = st.radio(
        "Choose how to answer",
        ["Evidence only", "LLM summary"],
        help="Evidence-only mode makes no API calls. LLM mode uses the provider/key you enter.",
    )
    llm_config = None
    if mode == "LLM summary":
        provider = st.selectbox("Provider", ["groq", "gemini", "openrouter"])
        defaults = {
            "groq": ("GROQ_API_KEY", "openai/gpt-oss-20b"),
            "gemini": ("GEMINI_API_KEY", "gemini-3.8-flash"),
            "openrouter": ("OPENROUTER_API_KEY", "openrouter/free"),
        }
        env_name, default_model = defaults[provider]
        api_key = st.text_input(f"{provider.title()} API key", type="password")
        model = st.text_input("Model ID", value=default_model)
        if api_key:
            llm_config = {"provider": provider, "model": model, "api_key": api_key}
        st.caption("The key stays in this browser session and is sent to the provider you selected.")
        st.caption("Check your provider's pricing and data-use terms before sending prompts.")

    st.divider()
    st.caption("Evidence is from a small, self-selected public review sample; it does not represent all listeners.")

question = st.text_area(
    "What do you want to learn from the reviews?",
    placeholder="Example: What do reviewers say about unexpected payments and refund support?",
    height=90,
)

if st.button("Search reviews", type="primary", disabled=not question.strip()):
    if mode == "LLM summary" and llm_config is None:
        st.error("Enter an API key for the selected provider, or choose Evidence only.")
    else:
        try:
            with st.spinner("Finding and checking relevant reviews…"):
                result = rag.run(
                    question.strip(),
                    top_k=8,
                    llm_config=llm_config,
                    use_llm=(mode == "LLM summary"),
                    retriever=get_retriever(),
                    reranker=get_reranker(),
                )
            st.subheader("Summary")
            st.write(result["answer"])
            if result["rewrite_method"] == "offline-rule-based":
                st.caption("Search terms were expanded with simple rules; no LLM query rewrite was used.")
            elif result["rewrite_method"] == "llm":
                with st.expander("LLM search rewrites"):
                    for query in result["rewritten_queries"]:
                        st.write(f"- {query}")
            if result["answer_method"] != "LLM synthesis with citation audit":
                st.info(f"Answer mode: {result['answer_method']}")

            st.subheader(f"Supporting reviews ({len(result['evidence'])})")
            for item in result["evidence"]:
                score = item["rerank_score"]
                score_text = f" · rank score {score:.2f}" if score is not None else ""
                label = (
                    f"[{item['citation']}] · {item['rating']}/5 · {item['date']}{score_text}"
                )
                with st.expander(label):
                    st.write(item["text"])
                    st.caption(
                        f"Embedding similarity: {item['embedding_score']:.3f}. "
                        "Ranker scores are relative scores, not probabilities."
                    )
        except Exception as exc:
            st.error(f"The search could not complete: {exc}")

st.divider()
st.caption(
    "Prototype for review triage. Public reviews are self-selected, may contain inaccurate "
    "accounts, and are not evidence of verified billing or product events."
)
