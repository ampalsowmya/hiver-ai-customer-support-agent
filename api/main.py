from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.pipeline import run_agent


app = FastAPI(
    title="Hiver AI Customer Support Agent",
    description="AI-powered customer support triage using intent classification and historical case retrieval.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TriageRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="Customer support message",
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of historical cases to retrieve",
    )


@app.get("/")
def root():
    return {
        "service": "hiver-ai-support-agent",
        "endpoints": {
            "health": "/health",
            "triage": "/triage",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "hiver-ai-support-agent",
    }


@app.post("/triage")
def triage(request: TriageRequest):

    result = run_agent(
        request.message,
        top_k=request.top_k,
    )

    return {
        "query": result["query"],
        "intent": result["intent"],
        "classifier_confidence": result["classifier_confidence"],
        "classifier_source": result.get("classifier_source"),
        "retrieval_similarity": result["retrieval_similarity"],
        "intent_compatibility": result.get("intent_compatibility"),
        "combined_score": result.get("combined_score"),
        "action": result["action"],
        "decision_reason": result["decision_reason"],
        "response": result["response"],
        "grounded": result["grounded"],
        "needs_escalation": result["needs_escalation"],
        "historical_cases": result.get("retrieved_cases", []),
    }