from services.rag.models import RewriteResult


def rewrite_query(
    query: str,
    history: list,
) -> RewriteResult:

    if not should_rewrite(query, history):

        return RewriteResult(
            original_query=query,
            retrieval_query=query,
            original_length=len(query),
            rewritten_length=len(query),
        )

    prompt = langfuse.get_prompt(
        "muallim-rewrite_query-prompt",
        type="chat",
    )

    compiled_prompt = prompt.compile(
        query=query,
        history=history,
    )

    llm = _get_llm_for_query_rewriter()

    try:

        response = llm.invoke(compiled_prompt)

        return validate_rewrite(
            original=query,
            rewrite=response.content,
        )

    except Exception:

        logger.exception("Query rewriting failed.")

        return RewriteResult(
            original_query=query,
            retrieval_query=query,
            used_fallback=True,
            fallback_reason="llm_exception",
            original_length=len(query),
            rewritten_length=len(query),
        )