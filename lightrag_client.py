import os
import asyncio
import inspect
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
from PyPDF2 import PdfReader

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./data"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "lightrag_compatible_demo.log")
    )

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-chat"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
        **kwargs,
    )


async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: openai_embed(
                texts,
                model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
                base_url=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
                api_key=os.getenv("EMBEDDING_BINDING_API_KEY", "your-api-key-here"),
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def query_rag(rag, mode, query_text):
    """执行查询并返回结果"""
    try:
        resp = await rag.aquery(
            query_text,
            param=QueryParam(mode=mode, stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)
    except Exception as e:
        print(f"查询过程中发生错误: {e}")


async def interactive_query(rag):
    """交互式查询界面"""
    mode_mapping = {
        "1": "naive",
        "2": "local",
        "3": "global",
        "4": "hybrid"
    }

    mode_description = {
        "1": "Naive模式 - 基础向量检索",
        "2": "Local模式 - 基于局部上下文的检索",
        "3": "Global模式 - 基于全局知识的检索",
        "4": "Hybrid模式 - 混合检索"
    }

    print("\n=====================")
    print("LightRAG 交互式查询")
    print("=====================")

    while True:
        print("\n请选择查询模式:")
        for key, desc in mode_description.items():
            print(f"{key}. {desc}")
        print("0. 退出程序")

        choice = input("\n请输入数字选择模式 (0-4): ").strip()

        if choice == "0":
            print("程序退出。")
            break
        elif choice in mode_mapping:
            mode = mode_mapping[choice]
            query_text = input(f"\n请输入您的查询问题 (当前模式: {mode}): ").strip()
            if query_text:
                print(f"\n正在执行 {mode} 模式查询...")
                print("-" * 30)
                await query_rag(rag, mode, query_text)
                print("\n" + "-" * 30)
            else:
                print("查询内容不能为空，请重新输入。")
        else:
            print("无效选择，请输入 0-4 之间的数字。")


async def main():
    try:
        is_first_run = 0
        if is_first_run:
            # Clear old data files
            files_to_delete = [
                "graph_chunk_entity_relation.graphml",
                "kv_store_doc_status.json",
                "kv_store_full_docs.json",
                "kv_store_text_chunks.json",
                "vdb_chunks.json",
                "vdb_entities.json",
                "vdb_relationships.json",
            ]

            for file in files_to_delete:
                file_path = os.path.join(WORKING_DIR, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

        if is_first_run:
            # with open("../dataset/K系列调试作业指导书.pdf", "r", encoding="utf-8") as f:
            #     await rag.ainsert(f.read())
            pdf_path = "../dataset/K系列调试作业指导书.pdf"
            reader = PdfReader(pdf_path)
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
            await rag.ainsert(text_content)

        # 启动交互式查询
        await interactive_query(rag)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
