"""
tutor_agent.py
--------------
LangGraph-powered autonomous tutoring agent for EduPulse.
Adapts coaching style based on the learner's ML-derived tier and
fetches targeted resources from the knowledge base via RAG.
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from knowledge_base import KnowledgeHub

load_dotenv()

# ---------------------------------------------------------------------------
# Knowledge hub (singleton)
# ---------------------------------------------------------------------------

_hub = KnowledgeHub()
_hub.initialise()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def fetch_study_material(topic: str) -> str:
    """
    Searches EduPulse's curated knowledge base for learning resources
    related to the given topic or concept.
    """
    resources = _hub.find_resources(topic)
    if not resources:
        return "No relevant resources found in the knowledge base for that topic."
    lines = [f"• {r['content']}  [Source: {r['source']}]" for r in resources]
    return "📚 Relevant study materials:\n" + "\n".join(lines)


TOOLS = [fetch_study_material]


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------


def _build_llm() -> ChatGroq:
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key == "your_groq_api_key_here":
        print("[TutorAgent] ⚠️  GROQ_API_KEY not set — please add it to your .env file.")
    return ChatGroq(temperature=0.6, model_name="llama-3.3-70b-versatile")


_llm = _build_llm()
_llm_with_tools = _llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class TutorState(TypedDict):
    messages: Annotated[list, add_messages]
    learner_tier: str
    pass_probability: float


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

_TIER_GUIDANCE = {
    "Struggling": (
        "Be warm, patient, and highly encouraging. Break down every concept into "
        "small steps. Celebrate small wins. Avoid jargon. Offer analogies."
    ),
    "Developing": (
        "Keep a balanced, motivating tone. Reinforce core concepts while "
        "introducing slightly more advanced material. Give practical examples."
    ),
    "Excelling": (
        "Be intellectually engaging. Challenge the student with edge cases, "
        "deeper theory, and real-world problem-solving scenarios."
    ),
}


def tutor_node(state: TutorState) -> dict:
    """Primary reasoning node — invokes the LLM (with tool access)."""
    msgs = state["messages"]

    # Prepend a system message once per session
    if not any(isinstance(m, SystemMessage) for m in msgs):
        tier = state.get("learner_tier", "Developing")
        prob = state.get("pass_probability", 0.5)
        guidance = _TIER_GUIDANCE.get(tier, _TIER_GUIDANCE["Developing"])

        system = SystemMessage(
            content=(
                f"You are EduPulse — an empathetic, expert AI tutor. "
                f"The learner you are coaching belongs to the '{tier}' performance tier "
                f"with an estimated pass probability of {prob:.0%}. "
                f"Coaching guidance: {guidance} "
                f"Always respond in a structured, easy-to-read way. "
                f"Use the fetch_study_material tool whenever the student asks about a "
                f"concept or topic that would benefit from a concrete reference."
            )
        )
        msgs = [system] + list(msgs)

    response = _llm_with_tools.invoke(msgs)
    return {"messages": [response]}


def tool_execution_node(state: TutorState) -> dict:
    """Executes any tool calls requested by the LLM."""
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return {"messages": []}

    results = []
    for call in last.tool_calls:
        if call["name"] == "fetch_study_material":
            output = fetch_study_material.invoke(call["args"])
            results.append(ToolMessage(content=output, tool_call_id=call["id"]))
    return {"messages": results}


def routing_decision(state: TutorState) -> str:
    """Routes to tool execution or ends the turn."""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "execute_tools"
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_graph = StateGraph(TutorState)
_graph.add_node("tutor", tutor_node)
_graph.add_node("execute_tools", tool_execution_node)

_graph.add_edge(START, "tutor")
_graph.add_conditional_edges("tutor", routing_decision, ["execute_tools", END])
_graph.add_edge("execute_tools", "tutor")

tutor_executor = _graph.compile()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def chat(
    user_message: str,
    learner_tier: str,
    pass_probability: float,
    history: list | None = None,
) -> list:
    """
    Send a message to the tutor agent and return the updated message list.

    Args:
        user_message: The student's latest input.
        learner_tier: ML-derived tier label ("Struggling", "Developing", "Excelling").
        pass_probability: Predicted probability of passing (0–1).
        history: Prior message list to maintain session continuity.

    Returns:
        Updated list of LangChain message objects.
    """
    history = history or []
    initial_state: TutorState = {
        "messages": history + [HumanMessage(content=user_message)],
        "learner_tier": learner_tier,
        "pass_probability": pass_probability,
    }
    result = tutor_executor.invoke(initial_state)
    return result["messages"]
