from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

CONNECTION_STRING = "postgresql://lovemeplease@localhost:5432/embedding"
COLLECTION_NAME = "long_paragraphs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TXT_PATH = "sample.txt"

with open(TXT_PATH, "r") as f:
    content = f.read()

chunk_size = 500
chunk_overlap = 50
splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = splitter.split_text(content)

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokens = encoding.encode(chunks[0])
token_count = len(tokens)

print("Chunk Size", chunk_size)
print("Chunk Overlap", chunk_overlap)
print("Chunk token", token_count)


documents = [
    Document(page_content=chunk, metadata={"source": TXT_PATH, "chunk": i})
    for i, chunk in enumerate(chunks)
]

embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

existing = vectorstore.similarity_search("artificial intelligence", k=1)

if existing and any(e.page_content in content for e in existing):
    print("✅ At least one similar chunk already embedded. Skipping insertion.")
else:
    print(f"📥 Inserting {len(documents)} chunks...")
    PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME
    )
    print("✅ Insertion complete.")
