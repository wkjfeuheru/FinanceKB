import asyncio

import pytest

from src.agent.planning import QueryPlan, Subproblem
from src.retrieve.pipeline import (
    RetrievalDependencies,
    RetrievalPolicy,
    RetrievalRequest,
    execute_retrieval,
)


@pytest.mark.asyncio
async def test_planning_and_filter_inference_run_concurrently():
    planning_started = asyncio.Event()
    inference_started = asyncio.Event()

    async def planner(query, *args):
        planning_started.set()
        await inference_started.wait()
        return QueryPlan(needs_decomposition=False, subproblems=[Subproblem(id="q1", query=query)])

    async def infer(query):
        inference_started.set()
        await planning_started.wait()
        return {}

    deps = RetrievalDependencies(
        planner=planner,
        infer_filters=infer,
        search=lambda *args: {"docs": [], "diagnostics": {}},
        prepare_context=lambda docs: {"context_docs": docs, "sources": [], "context_text": ""},
    )
    result = await asyncio.wait_for(
        execute_retrieval(
            RetrievalRequest("主营业务"),
            RetrievalPolicy(grade=False),
            deps,
        ),
        timeout=1,
    )
    assert result["stop_reason"] == "retrieval_only"


@pytest.mark.asyncio
async def test_failed_planning_cancels_pending_async_inference():
    inference_started = asyncio.Event()
    inference_cancelled = asyncio.Event()

    async def planner(*args):
        await inference_started.wait()
        raise ValueError("规划失败")

    async def infer(*args):
        inference_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            inference_cancelled.set()

    with pytest.raises(ValueError, match="规划失败"):
        await execute_retrieval(
            RetrievalRequest("主营业务"),
            RetrievalPolicy(grade=False),
            RetrievalDependencies(planner=planner, infer_filters=infer),
        )
    assert inference_cancelled.is_set()
