# NeuraDynamics AI Pipeline

A robust, agentic AI pipeline demonstrating Retrieval-Augmented Generation (RAG) and API integration using LangChain, LangGraph, and Streamlit.

## Features

- **Agentic Routing**: Uses LangGraph to intelligently route user queries to either a Weather API or a Document QA system.
- **Weather Integration**: real-time weather data fetching via OpenWeatherMap API.
- **RAG Capability**: PDF ingestion, embedding generation (OpenAI), and vector storage (Qdrant) for accurate document questioning.
- **MVC Architecture**: Strict separation of concerns (Model-View-Controller).
- **Evaluation**: Integration hooks for LangSmith tracing and evaluation.
- **UI**: Modern, responsive Streamlit interface.

## Architecture

The project follows a strict MVC pattern:

- **Models** (`src/models`): Handle data logic (Weather API, Qdrant).
- **Services** (`src/services`): encapsulating LLM business logic and external service calls.
- **Controllers** (`src/controllers`): Orchestrate the application flow. `WorkflowController` manages graph steps, `MainController` handles UI-backend interaction.
- **Graph** (`src/graph`): Defines the LangGraph state machine.
- **Views** (`views`): Streamlit UI layer.

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd neura-dynamics
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   - Copy `.env.example` to `.env`.
   - Fill in your API keys:
     - `OPENWEATHER_API_KEY`: From OpenWeatherMap.
     - `OPENAI_API_KEY`: From OpenAI.
     - `LANGCHAIN_API_KEY`: For LangSmith tracing.

4. **Run the Application**:
   ```bash
   streamlit run views/app.py
   ```

## Testing

Run unit tests using pytest:

```bash
pytest tests/
```

## LangSmith Evaluation

To view traces and evaluations:
1. Ensure `LANGCHAIN_TRACING_V2=true` is set in `.env`.
2. Interact with the Agent in the UI.
3. Visit [LangSmith](https://smith.langchain.com/) to view traces for the project `neura-dynamics-demo`.

## Structure

```
neura-dynamics/
├── src/
│   ├── models/       # Data Access Layer
│   ├── services/     # Logic & LLM Layer
│   ├── controllers/  # Application Logic & Orchestration
│   ├── graph/        # LangGraph Workflow Definitions
│   └── utils/        # Configuration
├── views/            # Streamlit UI
├── tests/            # Unit Tests
└── data/             # Local storage for Vector DB
```
