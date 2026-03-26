# iPOSpays Knowledge Base — n8n Data Pipeline(build for kaveri comapany chatbot referance)

## Overview

This n8n workflow implements a **RAG (Retrieval-Augmented Generation) data pipeline** that:

1. Scrapes content from iPOSpays websites on a schedule or via webhook
2. Detects content changes using SHA-256 hashing (avoids re-processing unchanged pages)
3. Chunks text using n8n's built-in **Recursive Character Text Splitter** (~1600 chars with 200 overlap)
4. Generates embeddings via n8n's built-in **Embeddings Google Gemini** node (`text-embedding-004`)
5. Inserts documents into **Qdrant** via n8n's built-in **Qdrant Vector Store** node
6. Deletes stale/outdated chunks automatically

> **Note:** This workflow uses n8n's latest **LangChain cluster nodes** — no raw HTTP Request nodes for embeddings or vector store operations (except delete, which has no built-in equivalent).

## Target URLs

| URL | Purpose |
|-----|---------|
| `https://releases.ipospays.com/ipospays` | Release notes |
| `https://knowledge.ipospays.com/` | Knowledge base |
| `https://docs.ipospays.com/payment-terminals-integrations` | Payment terminal integration docs |

## Workflow Architecture

### Node Types Used

| Node | Type | Purpose |
|------|------|---------|
| Scheduled Trigger | `n8n-nodes-base.scheduleTrigger` | Runs every 6 hours |
| Webhook Trigger | `n8n-nodes-base.webhook` | Manual/API trigger |
| Define Target URLs | `n8n-nodes-base.set` | Comma-separated URL list |
| Split URLs to Items | `n8n-nodes-base.code` | Splits URLs into individual items |
| Fetch Website Data | `n8n-nodes-base.httpRequest` | Fetches page HTML |
| Extract Text & Hash | `n8n-nodes-base.code` | HTML→text + SHA-256 hash |
| Detect Changes | `n8n-nodes-base.code` | Compares with stored hash (static data) |
| Filter Changed Only | `n8n-nodes-base.if` | Routes changed vs unchanged |
| **Qdrant Vector Store** | `@n8n/n8n-nodes-langchain.vectorStoreQdrant` | **Insert Documents** mode — built-in LangChain node |
| **Embeddings Google Gemini** | `@n8n/n8n-nodes-langchain.embeddingsGoogleGemini` | Sub-node providing embeddings to Qdrant Vector Store |
| **Default Data Loader** | `@n8n/n8n-nodes-langchain.documentDefaultDataLoader` | Sub-node loading JSON data with metadata |
| **Recursive Character Text Splitter** | `@n8n/n8n-nodes-langchain.textSplitterRecursiveCharacterTextSplitter` | Sub-node for chunking (1600 chars, 200 overlap) |
| Prepare Delete Old Chunks | `n8n-nodes-base.code` | Builds Qdrant delete filter |
| Delete Old Vectors (Qdrant) | `n8n-nodes-base.httpRequest` | HTTP delete (no built-in delete op) |
| Log Skipped | `n8n-nodes-base.code` | Logs unchanged pages |
| Pipeline Summary | `n8n-nodes-base.code` | Final execution summary |

### LangChain Cluster Node Wiring

The **Qdrant Vector Store** node is a LangChain "root" cluster node. Sub-nodes connect to it via special AI connection types:

```
┌─────────────────────────────────┐
│  Qdrant Vector Store            │  (root cluster node, mode: Insert Documents)
│  collection: ipospays-knowledge │
└──────┬──────────┬───────────────┘
       │          │
  ai_embedding  ai_document
       │          │
       ▼          ▼
┌──────────┐  ┌──────────────────────┐
│ Embeddings│  │ Default Data Loader  │  (metadata: source, title, lastUpdated,
│ Google    │  │ (dataType: json)     │   contentHash, versionId)
│ Gemini    │  └──────────┬───────────┘
└──────────┘         ai_textSplitter
                          │
                          ▼
                 ┌─────────────────────┐
                 │ Recursive Character  │  (chunkSize: 1600, chunkOverlap: 200)
                 │ Text Splitter        │
                 └─────────────────────┘
```

### Full Pipeline Flow

```
Scheduled Trigger ──┐
                    ├──▶ Define Target URLs ──▶ Split URLs ──▶ Fetch ──▶ Extract & Hash
Webhook Trigger ────┘                                                        │
                                                                             ▼
                                                              Detect Changes (Hash Compare)
                                                                             │
                                                                    Filter Changed Only
                                                                     │              │
                                                                  TRUE           FALSE
                                                                     │              │
                                                        ┌────────────┤              ▼
                                                        │            │     Log Skipped
                                                        ▼            ▼
                                            ┌──────────────┐  Prepare Delete Old Chunks
                                            │ Qdrant Vector │         │
                                            │ Store (Insert)│         ▼
                                            │  + Gemini Emb │  Delete Old Vectors (HTTP)
                                            │  + Data Loader│
                                            │  + Splitter   │
                                            └───────┬───────┘
                                                    ▼
                                              Pipeline Summary
```

## Setup Instructions

### 1. Import the Workflow

1. Open your n8n instance
2. Go to **Workflows** → **Import from File**
3. Select `n8n-data-pipeline.json`

### 2. Configure Credentials

You need to set up **two credentials** in n8n:

#### Google Gemini (PaLM) API Key
- **Type**: Google PaLM API (`googlePalmApi`)
- **API Key**: Your Google AI Studio API key
- Used by: **Embeddings Google Gemini** sub-node
- Get a key at: [Google AI Studio](https://aistudio.google.com/apikey)

#### Qdrant API
- **Type**: Qdrant API (`qdrantApi`)
- **API Key**: Your Qdrant API key
- **URL**: Your Qdrant instance URL (e.g., `https://your-cluster.qdrant.io`)
- Used by: **Qdrant Vector Store** node

### 3. Configure Environment Variables

Set these in your n8n environment (Settings → Variables, or via `.env`) — only needed for the **Delete Old Vectors** HTTP node:

| Variable | Description | Example |
|----------|-------------|---------|
| `QDRANT_URL` | Full Qdrant base URL | `https://your-cluster.qdrant.io` or `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key | `your-qdrant-api-key` |

### 4. Qdrant Collection Setup

Create a Qdrant collection with:
- **Collection name**: `ipospays-knowledge`
- **Dimensions**: `768` (for Google Gemini `text-embedding-004`)
- **Distance**: `Cosine`

```bash
curl -X PUT 'http://localhost:6333/collections/ipospays-knowledge' \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    }
  }'
```

### 5. Update Credential IDs

In the workflow JSON, replace this placeholder with your actual Qdrant credential ID:
- `REPLACE_WITH_QDRANT_CREDENTIAL_ID` → your Qdrant credential ID

The Google Gemini credential is already set (ID: `4QVY5uthDryLbGRW`). Update it if yours is different.

You can find credential IDs in n8n under **Settings → Credentials**.

## Document Metadata Schema

Each document stored in Qdrant includes this metadata (set in the Default Data Loader node):

```json
{
  "source": "https://releases.ipospays.com/ipospays",
  "title": "iPOSpays Releases",
  "lastUpdated": "2026-02-22T10:00:00.000Z",
  "contentHash": "a1b2c3d4...",
  "versionId": "a1b2c3d4e5f6"
}
```

The text content is stored as the document's `pageContent` by the Default Data Loader.

## Change Detection

The workflow uses **SHA-256 content hashing** stored in n8n's workflow static data:

- On each run, fetched page text is hashed
- Hash is compared against the stored hash for that URL
- If unchanged → page is **skipped** (no embedding/insert)
- If changed → new documents are **inserted** and old vectors are **deleted**
- Version IDs (first 12 chars of hash) prevent deleting freshly inserted chunks

## Customization

| Setting | Location | Default |
|---------|----------|---------|
| Schedule frequency | `Scheduled Trigger` node | Every 6 hours |
| Target URLs | `Define Target URLs` node | 3 iPOSpays URLs |
| Chunk size | `Recursive Character Text Splitter` node | 1600 chars |
| Chunk overlap | `Recursive Character Text Splitter` node | 200 chars |
| Embedding model | `Embeddings Google Gemini` node | `text-embedding-004` (768 dims) |
| Vector DB collection | `Qdrant Vector Store` node | `ipospays-knowledge` |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 403/401 on fetch | Some pages may need auth headers — add to HTTP Request node |
| Empty text extraction | Site uses heavy JS rendering — switch to Puppeteer node |
| Embedding rate limits | Add a **Wait** node before the Qdrant Vector Store node |
| Large pages timeout | Increase `timeout` in Fetch Website Data node options |
| Qdrant connection error | Verify `qdrantApi` credential URL and API key in n8n |
| Delete not working | Check `QDRANT_URL` and `QDRANT_API_KEY` env vars are set |
