"""
Example usage of the LiteLLM-backed ChatClient.
This script does not run automatically here; it's for reference.
"""
from cuda_agent_core.llm import ChatClient, ChatMessage


def main() -> None:
    client = ChatClient()
    resp = client.chat([
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Say hello in 5 words."),
    ])
    print("Model:", resp.model)
    print("Latency (ms):", resp.latency_ms)
    print("Tokens:", resp.total_tokens)
    print("---\n" + resp.content)


if __name__ == "__main__":
    main()

