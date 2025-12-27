import asyncio
import os

import uvicorn
from dbos import DBOS, DBOSConfig
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from genai.models import AGENT_STATUS, AgentStartRequest, AgentStatus
from genai.workflows import agentic_research_workflow
from rich.console import Console

console = Console()

config: DBOSConfig = {
    "name": "hacker-news-agent",
    "system_database_url": os.environ.get("DBOS_SYSTEM_DATABASE_URL"),
}
DBOS(config=config)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/agents")
def start_agent(request: AgentStartRequest):
    # Start a durable agent in the background
    DBOS.start_workflow(agentic_research_workflow, request.topic)
    return {"ok": True}


@app.get("/agents", response_model=list[AgentStatus])
async def list_agents():
    # List all active agents and retrieve their statuses
    agent_workflows = await DBOS.list_workflows_async(
        name=agentic_research_workflow.__qualname__,
        sort_desc=True,
    )
    statuses: list[AgentStatus] = await asyncio.gather(
        *[DBOS.get_event_async(w.workflow_id, AGENT_STATUS) for w in agent_workflows]
    )
    for workflow, status in zip(agent_workflows, statuses):
        status.status = workflow.status
        status.agent_id = workflow.workflow_id
    return statuses


DBOS.launch()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
