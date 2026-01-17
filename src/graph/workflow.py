from langgraph.graph import StateGraph, END
from src.controllers.workflow_controller import WorkflowController
from src.graph.state import AgentState

def create_workflow(controller=None):
    """
    Creates and compiles the LangGraph workflow.
    """
    if controller is None:
        controller = WorkflowController()
    
    workflow = StateGraph(AgentState)
    
    # Define Nodes
    # Note: The controller methods match the signature (state: AgentState) -> dict which updates the state.
    workflow.add_node("classify", controller.determine_intent)
    workflow.add_node("handle_weather", controller.handle_weather)
    workflow.add_node("handle_rag", controller.handle_rag)
    
    # Define Edges
    workflow.set_entry_point("classify")
    
    def route_decision(state: AgentState):
        # Returns the next node name based on intent
        intent = state.get("intent")
        if intent == "weather":
            return "handle_weather"
        else:
            return "handle_rag"
            
    workflow.add_conditional_edges(
        "classify",
        route_decision,
        {
            "handle_weather": "handle_weather",
            "handle_rag": "handle_rag"
        }
    )
    
    workflow.add_edge("handle_weather", END)
    workflow.add_edge("handle_rag", END)
    
    return workflow.compile()
