"""
Game Theory Decision Analyzer — unified FastAPI app.

Supports local models (Gemma via Ollama) and OpenAI behind one provider layer,
uses structured outputs for reliable JSON, and computes expected values by
backward induction. See models.py / game_theory.py / providers.py.
"""
from __future__ import annotations

import logging
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config import settings
from game_theory import process_analysis
from models import Analysis, PayoffMatrix, TreeNode, llm_schema
from prompts import SYSTEM_PROMPT, build_analysis_prompt
from providers import (
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    list_ollama_models,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(settings.log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, debug=settings.debug)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")


class RecomputeRequest(BaseModel):
    """Re-run the deterministic engine on user-adjusted assumptions (no LLM call)."""

    decision_tree: List[TreeNode] = []
    payoff_matrix: Optional[PayoffMatrix] = None
    risk_aversion: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="-1 risk-seeking, 0 risk-neutral, 1 strongly risk-averse",
    )


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    query: str
    provider: Literal["ollama", "openai"] = settings.default_provider
    model_name: Optional[str] = None
    ollama_url: str = settings.default_ollama_url


def _build_provider(req: AnalysisRequest) -> LLMProvider:
    if req.provider == "openai":
        if not settings.openai_api_key:
            raise HTTPException(
                status_code=400,
                detail="OpenAI provider selected but OPENAI_API_KEY is not configured on the server.",
            )
        model = req.model_name or settings.default_openai_model
        return OpenAIProvider(settings.openai_api_key, model, settings.temperature)

    model = req.model_name or settings.default_model_name
    return OllamaProvider(req.ollama_url, model, settings.temperature)


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/models")
async def get_models(ollama_url: str = settings.default_ollama_url):
    """List installed Ollama models (for the UI dropdown) and provider availability."""
    return {
        "ollama_models": list_ollama_models(ollama_url),
        "default_ollama_model": settings.default_model_name,
        "openai_available": bool(settings.openai_api_key),
        "default_openai_model": settings.default_openai_model,
        "default_provider": settings.default_provider,
    }


@app.post("/api/analyze", response_model=Analysis)
async def analyze_situation(req: AnalysisRequest):
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    logger.info("Analyze | provider=%s model=%s query=%s",
                req.provider, req.model_name, req.query[:80])

    provider = _build_provider(req)

    try:
        raw = provider.generate_json(
            SYSTEM_PROMPT, build_analysis_prompt(req.query), llm_schema()
        )
    except ProviderError as e:
        logger.error("Provider error: %s", e)
        raise HTTPException(status_code=502, detail=str(e))

    try:
        analysis = Analysis.model_validate(raw)
    except ValidationError as e:
        logger.error("Schema validation failed: %s", e)
        raise HTTPException(
            status_code=502,
            detail="The model's response did not match the expected structure. Try again or use a larger model.",
        )

    analysis = process_analysis(analysis)
    logger.info("Analysis complete | optimal=%s EV=%s",
                analysis.optimal_decision, analysis.optimal_expected_value)
    return analysis


@app.post("/api/recompute", response_model=Analysis)
async def recompute(req: RecomputeRequest):
    """Recompute expected values and Nash equilibria from edited assumptions.

    Purely deterministic — this runs the same engine as /api/analyze but never
    calls a model, so sensitivity analysis is instant.
    """
    analysis = Analysis(
        decision_tree=req.decision_tree,
        payoff_matrix=req.payoff_matrix,
    )
    return process_analysis(analysis, req.risk_aversion)


def main():
    logger.info("Starting %s on %s:%s", settings.app_name, settings.app_host, settings.app_port)
    logger.info("Default provider=%s model=%s", settings.default_provider, settings.default_model_name)
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
