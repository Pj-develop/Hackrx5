from qdrant_client import QdrantClient

client = QdrantClient(":memory:")  # Qdrant is running from RAM.
docs = ["Qdrant has a LangChain integration for chatbots.", "Qdrant has a LlamaIndex integration for agents."]
metadata = [
    {"source": "langchain-docs"},
    {"source": "llamaindex-docs"},
]
ids = [42, 2]
client.add(
    collection_name="test_collection",
    documents=docs,
    metadata=metadata,
    ids=ids
)
search_result = client.query(
    collection_name="test_collection",
    query_text="Which integration is best for agents?"
)
print(search_result)

