import pandas as pd

# ================= Load and Preprocess Data =================#
df_qa = pd.read_csv("./data/medical_qna_dataset.csv")
df_qa = df_qa.sample(500, random_state=0).reset_index(drop=True)


df_qa["combined_text"] = (
    "Question: "
    + df_qa["Question"].astype(str)
    + ". "
    + "Answer: "
    + df_qa["Answer"].astype(str)
    + ". "
    + "Type: "
    + df_qa["qtype"].astype(str)
    + ". "
)


df_md = pd.read_csv("./data/medical_device_manuals_dataset.csv")
df_md = df_md.sample(500, random_state=0).reset_index(drop=True)


df_md["combined_text"] = (
    "Device Name: "
    + df_md["Device_Name"].astype(str)
    + ". "
    + "Model: "
    + df_md["Model_Number"].astype(str)
    + ". "
    + "Manufacturer: "
    + df_md["Manufacturer"].astype(str)
    + ". "
    + "Indications: "
    + df_md["Indications_for_Use"].astype(str)
    + ". "
    + "Contraindications: "
    + df_md["Contraindications"].fillna("None").astype(str)
)

# ================= ChromaDB Persistent Client =================#
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection1 = client.get_or_create_collection(name="medical_qna")


collection1.add(
    documents=df_qa["combined_text"].tolist(), metadatas=df_qa.to_dict(orient="records"), ids=df_qa.index.astype(str).tolist()
)

collection2 = client.get_or_create_collection(name="medical_device_manual")

collection2.add(
    documents=df_md["combined_text"].tolist(),
    metadatas=df_md.to_dict(orient="records"),
    ids=df_md.index.astype(str).tolist(),
)

query = "what are the devices relevant to surgery"

results = collection2.query(query_texts=[query], n_results=3)
print(results)


# ================= LangChain with Ollama =================#


def get_llm_response(prompt: str) -> str:
    """Function to get response from LLM"""
    from langchain_community.llms import Ollama

    llm = Ollama(model="deepseek-r1:1.5b", timeout=120.0)
    response = llm.invoke(prompt)
    return response


prompt = "What is speciality of dunkirk"
response = get_llm_response(prompt)

print(response)


# ================= LangGraph Implementation =================#


from langgraph.graph import END, START, StateGraph


def retrieve_context_q_n_a(state):
    """Retrieve top documents from ChromaDB Collection 1 (Medical Q&A Data) based on query."""
    print("---RETRIEVING CONTEXT---")
    query = state["query"]
    results = collection1.query(query_texts=[query], n_results=3)
    context = "\n".join(results["documents"][0])
    state["context"] = context
    state["source"] = "Medical Q&A Collection"
    print(context)

    return state


def retrieve_context_medical_device(state):
    """Retrieve top documents from ChromaDB Collection 2 (Medical Device Manuals Data) based on query."""
    query = state["query"]
    results = collection2.query(query_texts=[query], n_results=3)
    context = "\n".join(results["documents"][0])
    state["context"] = context
    state["source"] = "Medical Device Manual"
    print(context)

    return state


def tavily_web_search(state):
    """Perform web search using the Tavily Search API."""
    print("--- Performing Tavily Web Search ---")
    tavily_search = TavilySearch(topic="general", max_results=1)
    query = state["query"]
    result_ = tavily_search.invoke({"query": query})
    state["context"] = result_["results"][0]["content"]

    return state


def router(state: GraphState) -> Literal["Retrieve_QnA", "Retrieve_Device", "Web_Search"]:
    """Agentic router: decides which retrieval method to use."""
    query = state["query"]

    # A lightweight decision LLM — you can replace this with GPT-4o-mini, etc.
    decision_prompt = f"""
    You are a routing agent. Based on the user query, decide where to look for information.

    Options:
    - Retrieve_QnA: if it's about general medical knowledge, symptoms, or treatment.
    - Retrieve_Device: if it's about medical devices, manuals, or instructions.
    - Web_Search: if it's about recent news, brand names, or external data.

    Query: "{query}"

    Respond ONLY with one of: Retrieve_QnA, Retrieve_Device, Web_Search
    """

    router_decision = get_llm_response(decision_prompt).strip()
    print(f"---ROUTER DECISION: {router_decision}---")

    print(router_decision)

    state["source"] = router_decision

    return state


def route_decision(state: GraphState) -> str:
    return state["source"]


def check_context_relevance(state: GraphState):
    """Determine whether the retrieved context is relevant or not."""
    print("---CONTEXT RELEVANCE CHECKER---")
    query = state["query"]
    context = state["context"]

    relevance_prompt = f"""
    Check the context below to see if the context is relevant to the user query or not.
    ####
    Context:
    {context}
    ####
    User Query: {query}

    Options:
    - Yes: if the context is relevant.
    - No: if the context is not relevant.

    Please answer with only 'Yes' or 'No'.
    """
    relevance_decision_value = get_llm_response(relevance_prompt).strip()
    print(f"---RELEVANCE DECISION: {relevance_decision_value}---")
    state["is_relevant"] = relevance_decision_value

    return state


def relevance_decision(state: GraphState) -> str:
    iteration_count = state.get("iteration_count", 0)
    iteration_count += 1
    state["iteration_count"] = iteration_count
    ## Limiting to max 3 iterations
    if iteration_count >= 3:
        print("---MAX ITERATIONS REACHED, FORCING 'Yes'---")
        state["is_relevant"] = "Yes"
    return state["is_relevant"]


def build_prompt(state):
    """Construct the RAG-style prompt."""
    print("---AUGMENT (BUILDING GENERATIVE PROMPT)---")
    query = state["query"]
    context = state["context"]

    prompt = f"""
    Answer the following question using the context below.
    Context:
    {context}
    Question: {query}
    please limit your answer in 50 words.
    """

    state["prompt"] = prompt
    print(prompt)
    return state


def call_llm(state):
    """Call your existing LLM function."""
    print("---GENERATE (CALLING LLM)---")
    prompt = state["prompt"]
    answer = get_llm_response(prompt)
    state["response"] = answer
    return state


# ================= LangGraph Workflow Definition =================#


class GraphState(TypedDict):
    query: str
    context: str
    prompt: str
    response: str
    source: str
    is_relevant: str
    iteration_count: str


workflow = StateGraph(GraphState)

workflow.add_node("Router", router)
workflow.add_node("Retrieve_QnA", retrieve_context_q_n_a)
workflow.add_node("Retrieve_Device", retrieve_context_medical_device)
workflow.add_node("Web_Search", tavily_web_search)
workflow.add_node("Relevance_Checker", check_context_relevance)
workflow.add_node("Augment", build_prompt)
workflow.add_node("Generate", call_llm)

workflow.add_edge(START, "Router")
workflow.add_conditional_edges(
    "Router",
    route_decision,
    {
        "Retrieve_QnA": "Retrieve_QnA",
        "Retrieve_Device": "Retrieve_Device",
        "Web_Search": "Web_Search",
    },
)
workflow.add_edge("Retrieve_QnA", "Relevance_Checker")
workflow.add_edge("Retrieve_Device", "Relevance_Checker")
workflow.add_edge("Web_Search", "Relevance_Checker")
workflow.add_conditional_edges(
    "Relevance_Checker",
    relevance_decision,
    {
        "Yes": "Augment",
        "No": "Web_Search",
    },
)
workflow.add_edge("Augment", "Generate")
workflow.add_edge("Generate", END)

agentic_rag = workflow.compile()
