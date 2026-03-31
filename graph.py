from langgraph.graph import StateGraph, START, END
from state import ResearchState
from planner_node import planner_node, route_to_researchers
from researcher_node import researcher_node
from critic_node import critic_node, route_after_critic
from writer_node import writer_node
from image_node import image_node


def _route_after_writer(state: ResearchState) -> str:
    """If images are enabled, go to image_node; otherwise go straight to END."""
    if state.get("images_enabled", False):
        return "image_node"
    return END


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner_node",    planner_node)
    graph.add_node("researcher_node", researcher_node)
    graph.add_node("critic_node",     critic_node)
    graph.add_node("writer_node",     writer_node)
    graph.add_node("image_node",      image_node)

    # START → planner
    graph.add_edge(START, "planner_node")

    # planner → N parallel researchers via Send()
    graph.add_conditional_edges(
        "planner_node",
        route_to_researchers,
        ["researcher_node"],
    )

    # researchers → critic
    graph.add_edge("researcher_node", "critic_node")

    # critic → retry planner OR proceed to writer
    graph.add_conditional_edges(
        "critic_node",
        route_after_critic,
        {
            "planner_node": "planner_node",
            "writer_node":  "writer_node",
        },
    )

    # writer → image_node (if enabled) OR END
    graph.add_conditional_edges(
        "writer_node",
        _route_after_writer,
        {
            "image_node": "image_node",
            END:           END,
        },
    )

    # image_node → END
    graph.add_edge("image_node", END)

    return graph.compile()


app = build_graph()