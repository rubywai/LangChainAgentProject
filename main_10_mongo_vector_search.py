from dotenv import load_dotenv
from pathlib import Path
import os
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

load_dotenv(dotenv_path=Path(__file__).parent / '.env')

# Configuration from environment - you provided these values
MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb+srv://cheatouser:<db_password>@cluster0.wmueyun.mongodb.net/?appName=Cluster0')
DB_NAME = os.environ.get('MONGO_DB_NAME', 'rab_test')
COLLECTION_NAME = os.environ.get('MONGO_COLLECTION', 'rab_test')
INDEX_NAME = os.environ.get('MONGO_INDEX_NAME', 'vector_index')

# Embedding model to use with Ollama
EMBED_MODEL = os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')

# Simple documents to index (similar to your real Chroma example)
DOCUMENTS = [
    Document(
        page_content="Flutter is a cross-platform mobile development framework created by Google. It uses Dart programming language and allows building apps for iOS, Android, and web from a single codebase. Flutter has hot reload, rich widgets, and excellent performance.",
        metadata={"course": "Flutter", "topic": "Mobile Development", "students": 150}
    ),
    Document(
        page_content="Kotlin is a modern programming language for Android development. It's officially supported by Google and offers null safety, coroutines, and concise syntax. Kotlin is 100% interoperable with Java.",
        metadata={"course": "Kotlin", "topic": "Android Development", "students": 120}
    ),
    Document(
        page_content="LangChain is a framework for developing applications powered by language models. It provides tools for chains, agents, memory management, and RAG (Retrieval Augmented Generation).",
        metadata={"course": "LangChain", "topic": "AI Development", "students": 80}
    )
]


def get_vector_store(client: MongoClient, embeddings: OllamaEmbeddings) -> MongoDBAtlasVectorSearch:
    """Create or get MongoDBAtlasVectorSearch instance.

    This uses the official langchain-mongodb integration which handles:
    - Vector embeddings automatically
    - Atlas Vector Search integration
    - Fallback to similarity search if Atlas search not available
    """
    collection = client[DB_NAME][COLLECTION_NAME]

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME,
        text_key="page_content",  # field name for document text
        embedding_key="embedding",  # field name for embeddings
    )

    return vector_store



def main():
    print("🚀 MongoDB Atlas Vector Search with LangChain")
    print("=" * 60)

    # Validate environment
    if not MONGODB_URI or '<db_password>' in MONGODB_URI:
        print("❌ Error: MONGODB_URI not properly set in .env file")
        print("   Please set MONGODB_URI with your actual MongoDB password")
        print("   Example: MONGODB_URI=mongodb+srv://user:password@cluster0...")
        return

    # Initialize
    print(f"\n📦 Connecting to MongoDB...")
    print(f"   Database: {DB_NAME}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Index: {INDEX_NAME}")

    client = MongoClient(MONGODB_URI)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # Get vector store
    vector_store = get_vector_store(client, embeddings)

    # Add documents
    print(f"\n📝 Adding {len(DOCUMENTS)} documents to MongoDB...")
    try:
        # MongoDBAtlasVectorSearch.add_documents handles embedding automatically
        ids = vector_store.add_documents(DOCUMENTS)
        print(f"✅ Successfully indexed {len(ids)} documents")

        # Verify what was stored
        collection = client[DB_NAME][COLLECTION_NAME]
        doc_count = collection.count_documents({})
        print(f"📊 Total documents in collection: {doc_count}")

        # Show a sample document structure
        sample = collection.find_one()
        if sample:
            print(f"📄 Sample document fields: {list(sample.keys())}")

    except Exception as e:
        print(f"⚠️  Error adding documents: {e}")
        print("   (Documents might already exist, continuing...)")
        import traceback
        traceback.print_exc()

    # Example queries
    queries = [
        "Tell me about mobile app development frameworks",
        "How many students enrolled in Kotlin course?",
        "What is LangChain used for?",
    ]

    print("\n" + "=" * 60)
    print("🔍 Running similarity searches:")
    print("=" * 60)

    for query in queries:
        print(f"\n📌 Query: {query}")
        print("-" * 60)

        try:
            # Try Atlas Vector Search first
            try:
                results = vector_store.similarity_search(query, k=2)
                if results:
                    print(f"   ✅ Using Atlas Vector Search")
                    for i, doc in enumerate(results, 1):
                        print(f"\n   Result {i}:")
                        print(f"   Course: {doc.metadata.get('course', 'N/A')}")
                        print(f"   Topic: {doc.metadata.get('topic', 'N/A')}")
                        print(f"   Students: {doc.metadata.get('students', 'N/A')}")
                        print(f"   Content: {doc.page_content[:150]}...")
                    continue
            except Exception as atlas_error:
                print(f"   ⚠️  Atlas Vector Search not available: {str(atlas_error)[:100]}")
                print(f"   🔄 Falling back to manual similarity search...")

            # Manual fallback: compute similarities locally
            import math
            query_embedding = embeddings.embed_query(query)

            # Get all documents from collection
            collection = client[DB_NAME][COLLECTION_NAME]
            all_docs = list(collection.find({}))

            if not all_docs:
                print("   No documents in collection")
                continue

            # Calculate cosine similarity for each
            scored_docs = []
            for doc_data in all_docs:
                if 'embedding' not in doc_data:
                    continue

                doc_embedding = doc_data['embedding']

                # Cosine similarity
                dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                mag_a = math.sqrt(sum(a * a for a in query_embedding))
                mag_b = math.sqrt(sum(b * b for b in doc_embedding))
                similarity = dot_product / (mag_a * mag_b) if (mag_a * mag_b) > 0 else 0.0

                scored_docs.append((similarity, doc_data))

            # Sort by similarity (highest first)
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            # Show top 2 results
            for i, (score, doc_data) in enumerate(scored_docs[:2], 1):
                print(f"\n   Result {i} (similarity: {score:.4f}):")
                # MongoDB stores metadata fields at top level
                metadata = doc_data.get('metadata', {})
                print(f"   Course: {doc_data.get('course', metadata.get('course', 'N/A'))}")
                print(f"   Topic: {doc_data.get('topic', metadata.get('topic', 'N/A'))}")
                print(f"   Students: {doc_data.get('students', metadata.get('students', 'N/A'))}")
                content = doc_data.get('page_content', doc_data.get('text', ''))
                print(f"   Content: {content[:150]}...")

        except Exception as e:
            print(f"   ❌ Search error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("\n💡 Note: For best performance, create a Vector Search Index in Atlas UI:")
    print(f"   - Database: {DB_NAME}")
    print(f"   - Collection: {COLLECTION_NAME}")
    print(f"   - Index name: {INDEX_NAME}")
    print("   - Field: embedding")
    print("   - Type: knnVector")

    client.close()


if __name__ == '__main__':
    main()
