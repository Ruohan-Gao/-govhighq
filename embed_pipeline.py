import os
import fitz  # PyMuPDF
import docx
import openai
import uuid
from tqdm import tqdm
from azure.storage.blob import ContainerClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration,
    SearchField, SearchableField, VectorSearchProfile
)
import io

from config import *

import json

PROCESSED_BLOBS_PATH = "tracking/processed_blobs.json"

def load_processed_blobs():
    try:
        blob = blob_client.get_blob_client(PROCESSED_BLOBS_PATH)
        content = blob.download_blob().readall()
        return set(json.loads(content))
    except Exception:
        return set()

def save_processed_blobs(processed_set):
    blob = blob_client.get_blob_client(PROCESSED_BLOBS_PATH)
    data = json.dumps(list(processed_set), indent=2)
    blob.upload_blob(data, overwrite=True)







# === Azure Clients ===
blob_client = ContainerClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER)

# # Updated OpenAI client initialization
# openai.api_type = "azure"
# openai.api_base = AZURE_OPENAI_ENDPOINT
# openai.api_version = AZURE_OPENAI_API_VERSION
# openai.api_key = AZURE_OPENAI_API_KEY

from azure.core.credentials import AzureKeyCredential

# Fixed credential initialization
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

# === Create Vector Index (Updated for newer SDK) ===
def create_vector_index():
    try:
        # Check if index already exists
        existing_indexes = [idx.name for idx in index_client.list_indexes()]
        if AZURE_SEARCH_INDEX_NAME in existing_indexes:
            print("Index already exists.")
            return
    except Exception as e:
        print(f"Error checking existing indexes: {e}")
        print("Proceeding to create new index...")

    # Updated vector search configuration for newer SDK
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="doc_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchField(
            name="embedding", 
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-config"
        )
    ]
    
    # Updated vector search configuration
    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="my-vector-config",
                algorithm_configuration_name="my-hnsw-config"
            )
        ],
        algorithms=[
            HnswAlgorithmConfiguration(name="my-hnsw-config")
        ]
    )
    
    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME, 
        fields=fields, 
        vector_search=vector_search
    )

    try:
        print(f"Creating index '{AZURE_SEARCH_INDEX_NAME}'...")
        index_client.create_index(index)
        print("Index created successfully!")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise

# === Text Extraction ===
def extract_text_from_blob(blob_name):
    try:
        blob = blob_client.get_blob_client(blob_name)
        content = blob.download_blob().readall()

        if blob_name.endswith(".pdf"):
            with fitz.open(stream=content, filetype="pdf") as doc:
                return "\n".join(page.get_text() for page in doc)
        elif blob_name.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return ""
    except Exception as e:
        print(f"Error extracting text from {blob_name}: {e}")
        return ""

# === Chunking ===
# def chunk_text(text, max_tokens=500):
#     paragraphs = text.split("\n")
#     chunks, chunk = [], ""
#     for p in paragraphs:
#         if len(chunk) + len(p) < max_tokens * 4:  # approx 4 chars per token
#             chunk += " " + p.strip()
#         else:
#             if chunk.strip():
#                 chunks.append(chunk.strip())
#             chunk = p
#     if chunk.strip():
#         chunks.append(chunk.strip())
#     return [c for c in chunks if c]  # Remove empty chunks


from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=750, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


# === Embedding ===
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def embed_text(text):
    try:
        response = client.embeddings.create(
            input=[text],
            model=AZURE_OPENAI_DEPLOYMENT
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        raise


# === Main Logic ===
def main():
    try:
        create_vector_index()

        print("Processing blobs...")
        processed_blobs = load_processed_blobs()
        updated_blobs = set()
        
        blobs = list(blob_client.list_blobs())

        for blob in tqdm(blobs):
            blob_name = blob.name
            if blob_name in processed_blobs or blob_name == PROCESSED_BLOBS_PATH:
                print(f"Skipping already processed file: {blob_name}")
                continue  # Skip already processed or tracking file

            doc_type = blob_name.split("/")[0] if "/" in blob_name else "unknown"
            print(f"Processing: {blob_name}")

            text = extract_text_from_blob(blob_name)
            if not text.strip():
                print(f"No text extracted from {blob_name}")
                continue

            chunks = chunk_text(text)
            if not chunks:
                print(f"No chunks created from {blob_name}")
                continue

            docs = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = embed_text(chunk)
                    docs.append({
                        "id": str(uuid.uuid4()),
                        "content": chunk,
                        "doc_type": doc_type,
                        "embedding": embedding
                    })
                except Exception as e:
                    print(f"Error processing chunk {i} from {blob_name}: {e}")
                    continue

            if docs:
                try:
                    search_client.upload_documents(documents=docs)
                    print(f"Uploaded {len(docs)} chunks from '{blob_name}'")
                    updated_blobs.add(blob_name)
                except Exception as e:
                    print(f"Error uploading documents from {blob_name}: {e}")
                    continue

        # Save updated list only if something new was processed
        if updated_blobs:
            processed_blobs.update(updated_blobs)
            save_processed_blobs(processed_blobs)

    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()


