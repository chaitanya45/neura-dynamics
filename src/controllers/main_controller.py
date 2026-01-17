from src.graph.workflow import create_workflow
from src.controllers.workflow_controller import WorkflowController

class MainController:
    """
    Main entry point for the application.
    Orchestrates the workflow and handles user interaction from the View.
    """
    def __init__(self):
        self.workflow_controller = WorkflowController()
        self.graph = create_workflow(self.workflow_controller)

    def handle_query(self, query: str):
        """
        Processes a user query through the LangGraph workflow.
        Returns the response text and the full state.
        """
        inputs = {"query": query}
        # Invoke the graph
        result = self.graph.invoke(inputs)
        return result.get("response", "No response generated."), result

    def upload_pdf(self, file_path: str):
        """
        Uploads and processes a PDF file via the RAG model.
        """
        return self.workflow_controller.rag_model.load_and_process_pdf(file_path)
