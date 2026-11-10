"""Make one minimal live API request using the provider configured in .env."""

from openai import APIStatusError, APITimeoutError, APIConnectionError

from agentic_rag import call_llm, resolve_llm_config


if __name__ == "__main__":
    try:
        provider, model, _, _ = resolve_llm_config()
        print(f"Using provider: {provider}; model: {model}")
        reply = call_llm(
            "Reply with exactly the text API_OK and nothing else.",
            "Confirm the configured API is reachable.",
        )
        print("Provider response:", reply)
    except APIStatusError as exc:
        if exc.status_code == 503:
            print(
                "The provider returned HTTP 503 (temporarily unavailable). "
                "The request was retried automatically. Wait a minute and run "
                "python check_api.py again, or try another model/provider."
            )
        elif exc.status_code == 429:
            print(
                "The provider returned HTTP 429 (rate limit/quota reached). "
                "Check the provider console's quota and retry after it resets."
            )
        else:
            print(f"The provider returned HTTP {exc.status_code}: {exc}")
        raise SystemExit(1)
    except (APITimeoutError, APIConnectionError) as exc:
        print(f"Could not connect to the configured API: {exc}")
        raise SystemExit(1)
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1)
