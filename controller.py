from fastapi import FastAPI, HTTPException

from execute_script import viewEmbed
from model.models import EmbedRequest, SummarizeRequest
from services.embed_service import embed_documents
from services.summarize_service import summarize_documents

app = FastAPI(title="Medical Embedding & Summarization API")

@app.post("/embed")
def embed_endpoint(request: EmbedRequest):
    try:
        embed_documents(request.files)
        return {"status": f"Embedded {len(request.files)} file(s)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.post("/summarize")
def summarize_endpoint(request: SummarizeRequest):
    try:
        summary = summarize_documents(request.files)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.get("/view-embed")
def view_embed():
    result = viewEmbed()
    return {
        "results": result
    }