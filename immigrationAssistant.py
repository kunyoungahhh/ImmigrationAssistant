from os import environ
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

OPENAI_API_KEY = environ["OPENAI_API_KEY"]

def build_agent() -> ReActAgent:
    llm = OpenAI(model="gpt-4")

    try:
        asylum_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./storage/asylum")
        )
        niw_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./storage/niw")
        )
    except Exception:
        asylum_docs = SimpleDirectoryReader(input_dir="./docs/asylum", recursive=True).load_data()
        niw_docs = SimpleDirectoryReader(input_dir="./docs/niw/", recursive=True).load_data()

        asylum_index = VectorStoreIndex.from_documents(asylum_docs, show_progress=True)
        niw_index = VectorStoreIndex.from_documents(niw_docs, show_progress=True)

        asylum_index.storage_context.persist(persist_dir="./storage/asylum")
        niw_index.storage_context.persist(persist_dir="./storage/niw")

    asylum_engine = asylum_index.as_query_engine(similarity_top_k=3, llm=llm)
    niw_engine = niw_index.as_query_engine(similarity_top_k=3, llm=llm)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=asylum_engine,
            metadata=ToolMetadata(
                name="asylum_docs",
                description="Provides information about asylum application process in the U.S."
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

    return ReActAgent.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=True,
        max_turns=10,
    )

def ask_question(agent: ReActAgent, user_input: str, topic: str) -> str:


    prompt = f"""Answer as detail as possible based on info below.

Tone: Inquisitive and insightful, with a gentle sense of respect for the subject matter.

Situation: {user_input}

Emphasis: Use specific form number and descriptive language so the user can fill out all the necessary information without help from an attorney. If users question is not related to the {topic}, refuse to answer it.

Example: You need the form I-589, your passport number, you need forms and documents from specific links. Give examples of proposed endeavors for NIW cases.

Format: Use number list or bullet list.

Also, include a checklist for the whole documents.

If the context does not contain the required information, search the web. Plus, based on what step is the user is at, provide prospective steps and forms the user needs in detail.

Exception: If user asks a specific question regarding a form, answer it briefly. For example, if the user asks what passport number to use if they have two passports, first try to answer based on the context. If not found, answer based on general practices. Do not answer questions that are not about the immigration law, revert user back to immigration law topic. Acknowledge the question but do not engage, instead say something like "I understand that you are interested about this topic, but I am designed to answer your questions about the immigration law" and then offer a new topic, keep it short - 2 sentences max.
"""
    return str(agent.chat(prompt)).replace('**','')
