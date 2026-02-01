import streamlit as st

st.set_page_config(
    page_title="Text2SQL Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 Text2SQL Agent Workspace")

st.markdown(
    """
Welcome to the Text2SQL Agent Workspace.

### Navigation
- **Chat**: Interact with the agent, view sub-agent thought processes, and get suggested questions.
- **Eval**: View the results of the latest evaluation runs and detailed judge reasoning.

Select a page from the sidebar to get started.

---
### Agent Control Flow
"""
)


# Mermaid Diagram helper
def mermaid(code: str):
    import streamlit.components.v1 as components

    components.html(
        f"""
        <pre class="mermaid">
            {code}
        </pre>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=600,
    )


mermaid(
    """
graph TD
    User((User)) -->|Controls| Root["Root Agent"]
    Root -->|Delegates| QA["Query & Answer Agent"]
    
    subgraph RecursiveLogic ["Recursive Logic"]
        QA -->|Controls| GetData["Get Data Agent (Loop)"]
        GetData -->|Step 1| GenSqlSeq["SQL Gen Sequential"]
        
        subgraph GenerationPhase ["Generation Phase"]
            GenSqlSeq -->|Check| Region["Region Check Agent"]
            GenSqlSeq -->|Generate| SqlGen["SQL Gen Agent"]
        end
        
        GetData -->|Step 2| CriticSeq["SQL Critic Sequential"]
        
        subgraph ValidationPhase ["Validation Phase"]
            CriticSeq -->|Verify| CheckSql["Check SQL Agent"]
            CriticSeq -->|Execute| ExecSql["Execute SQL Agent"]
            CriticSeq -->|Review| Critic["Critic Agent"]
        end
        
        Critic -.->|Feedback Loop| GetData
    end
    
    QA -->|Final Step| Ans["Answer Agent"]
    Ans -->|Response| User
    
    style User fill:#f9f,stroke:#333,stroke-width:2px,color:black
    style Root fill:#ccf,stroke:#333,stroke-width:2px,color:black
    style QA fill:#ccf,stroke:#333,stroke-width:2px,color:black
    style GetData fill:#ffc,stroke:#333,stroke-width:2px,color:black
    style Ans fill:#cfc,stroke:#333,stroke-width:2px,color:black
"""
)

st.sidebar.success("Select a page above.")
