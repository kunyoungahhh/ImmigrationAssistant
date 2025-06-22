from os import environ
from dotenv import load_dotenv
load_dotenv()
from tqdm.auto import tqdm

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.query_engine import SubQuestionQueryEngine




OPENAI_API_KEY = environ["OPENAI_API_KEY"]

SYSTEM_CONTEXT = """You are an immigration law assistant. 
You must always respond in English.

⚠️ You are **not allowed** to answer any question directly. EVER. 
All answers must be provided through a tool. 

You are part of a toolchain and do not have independent knowledge. You must ALWAYS call a tool to get an answer.

Even if the answer appears obvious or simple, you must **still use a tool** to validate and produce the final response.

If you are unsure which tool is appropriate, select the most closely related one and continue.

Never say 'I can answer this without a tool'. That is incorrect behavior.

Your goal is to assist with U.S. immigration matters such as asylum and NIW applications by consulting official document sources via tool usage.

Use clear bullet points, include form numbers where applicable, and do not answer any non-immigration questions.

Under no circumstances should you hallucinate or generate answers from your own memory.

Always rely on tool output for every response.
"""

class DebugQueryEngine:
    def __init__(self, engine, name=""):
        self.engine = engine
        self._name = name

    def query(self, q):
        response = self.engine.query(q)

        if hasattr(response, "source_nodes"):
            print(f"\n🧩 {self._name} - Relevant Chunks Used for Answer:")
            for i, node_with_score in enumerate(response.source_nodes):
                node = getattr(node_with_score, "node", node_with_score)
                page = node.metadata.get("page_label", "Unknown")
                print(f"\n--- Chunk {i + 1} (Page {page}) ---\n{node.text}\n")

        return response

    @property
    def callback_manager(self):
        return self.engine.callback_manager

    async def aquery(self, q):
        response = await self.engine.aquery(q)

        if hasattr(response, "source_nodes"):
            print(f"\n🧩 [ASYNC] {self._name} - Relevant Chunks Used for Answer:")
            for i, node_with_score in enumerate(response.source_nodes):
                node = getattr(node_with_score, "node", node_with_score)
                page = node.metadata.get("page_label", "Unknown")
                print(f"\n--- Chunk {i + 1} (Page {page}) ---\n{node.text}\n")

        return response

def build_agent() -> ReActAgent:
    llm = OpenAI(
        model="gpt-4o-mini",
        context=SYSTEM_CONTEXT
    )

    try:
        asylum_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./storage/asylum")
        )
        niw_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./storage/niw")
        )
    except Exception:
        print("📥 Reading documents from disk...")

        # Use tqdm manually for progress feedback
        raw_asylum_docs = SimpleDirectoryReader(input_dir="./docs/asylum", recursive=True).load_data()
        asylum_docs = list(tqdm(raw_asylum_docs, desc="📄 Asylum Docs Loaded"))

        raw_niw_docs = SimpleDirectoryReader(input_dir="./docs/niw/", recursive=True).load_data()
        niw_docs = list(tqdm(raw_niw_docs, desc="📄 NIW Docs Loaded"))

        print(f"✅ Loaded {len(asylum_docs)} asylum and {len(niw_docs)} NIW documents.")

        text_splitter = SentenceSplitter(
            chunk_size=512,  # adjust to match your LLM context window
            chunk_overlap=50,  # keeps some continuity between chunks
        )

        asylum_index = VectorStoreIndex.from_documents(asylum_docs, text_splitter=text_splitter,show_progress=True)
        niw_index = VectorStoreIndex.from_documents(niw_docs, text_splitter=text_splitter,show_progress=True)

        asylum_index.storage_context.persist(persist_dir="./storage/asylum")
        niw_index.storage_context.persist(persist_dir="./storage/niw")

    asylum_engine = DebugQueryEngine(
        asylum_index.as_query_engine(similarity_top_k=3, llm=llm, return_source=True),
        name="Asylum"
    )
    niw_engine = DebugQueryEngine(
        niw_index.as_query_engine(similarity_top_k=3, llm=llm, return_source=True),
        name="NIW"
    )

    query_engine = DebugQueryEngine(niw_index.as_query_engine())

    query_engine_tools = [
        QueryEngineTool(
            query_engine=asylum_engine,
            metadata=ToolMetadata(
                name="asylum_docs",
                description="Provides information about the asylum application process in the U.S."
            ),
        ),
        QueryEngineTool(
            query_engine=niw_engine,
            metadata=ToolMetadata(
                name="niw_docs",
                description="Provides information about NIW green card application process."
            ),
        ),
    ]

    combined_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[
            QueryEngineTool(
                query_engine=asylum_index.as_query_engine(similarity_top_k=3, llm=llm, return_source=True),
                metadata=ToolMetadata(name="asylum", description="Asylum-related queries"),
            ),
            QueryEngineTool(
                query_engine=niw_index.as_query_engine(similarity_top_k=3, llm=llm, return_source=True),
                metadata=ToolMetadata(name="niw", description="NIW green card queries"),
            ),
        ],
        llm=llm,
    )

    combined_engine = DebugQueryEngine(combined_engine, name="Combined")

    return ReActAgent.from_tools(
        tools=[
            QueryEngineTool(
                query_engine=combined_engine,
                metadata=ToolMetadata(
                    name="combined",
                    description="Handles all immigration queries from both asylum and NIW sources"
                )
            )
        ],
        llm=llm,
        verbose=True,
        tool_choice_mode="required",  # Force tool usage
    )

def ask_question(agent: ReActAgent, user_input: str, topic: str) -> str:


    prompt = f"""Answer this questions using the tool only.
User question (English only): {user_input}
"""
    return str(agent.chat(prompt)).replace('**','')