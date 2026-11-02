import os
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,
    check_every_n_seconds=0.1,
    max_bucket_size=1
    )

llm = init_chat_model(
    "groq:openai/gpt-oss-120b",
    rate_limiter=rate_limiter,
    max_tokens=1500,
    reasoning_effort="low"
    )

def make_retriever_tool_from_text(file,name,desc):
    docs=TextLoader(file,encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vs = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    def tool_func(query:str)->str:
        print(f"📚 Using tool: {name}")
        results = retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in results)
    
    return StructuredTool.from_function(
        func=tool_func,
        name=name,
        description=desc,
    )

web_search = TavilySearch(max_results=2)

internal_tool_1=make_retriever_tool_from_text("internal_docs.txt",
                                              "InternalResearchNotes",
                                              "Search internal research notes for experimental results")

def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b

math_agent=create_agent(
    model=llm,
    tools=[add,multiply,divide],
    system_prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent"

)

research_agent=create_agent(
    model=llm,
    tools=[web_search,internal_tool_1],
    system_prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent"
)

supervisor=create_supervisor(
    model=llm,
    agents=[research_agent,math_agent],
    prompt=(
         "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."

    ),
    add_handoff_back_messages=True,
    output_mode="last_message"
).compile()


try:
    png_bytes = supervisor.get_graph(xray=True).draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_bytes)
    print("Saved graph visualization to graph.png")
except Exception as e:
    print(f"Could not render graph image ({e}); printing mermaid source instead:\n")
    print(supervisor.get_graph(xray=True).draw_mermaid())
    
if __name__== "__main__":
    
    response = supervisor.invoke(
        {"messages": [HumanMessage(content="Write a detailed blog on transformer variants in production deployments")]},
        config={"recursion_limit": 12},
    )
    # Collect real answers from the research agent (skip tool calls and handoff notices)
    candidates = [
        m.content for m in response["messages"]
        if isinstance(m, AIMessage)
        and m.name == "research_agent"
        and m.content
        and not m.tool_calls
        and not m.content.startswith("Transferring back")
    ]
    final = max(candidates, key=len) if candidates else response["messages"][-1].content
    print("\n--- FINAL ANSWER ---\n")
    print(final)