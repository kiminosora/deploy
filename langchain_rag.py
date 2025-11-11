import os
from typing import List, Dict, Any

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import json


class LocalVLLMChatModel(Runnable):
    """
    一个简单的LangChain兼容的本地VLLM模型接口
    """

    def __init__(self, base_url: str = "http://localhost:23333", model_name: str = "Qwen3-4B-Instruct-2507"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def _call_api(self, messages: List[Dict[str, str]], stream: bool = False, max_tokens: int = 512):
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    def invoke(self, input, config=None, **kwargs):
        # 如果输入是字符串，将其转换为消息格式
        if isinstance(input, str):
            messages = [
                {"role": "user", "content": input}
            ]
        # 如果输入是字典，假设它包含消息
        elif isinstance(input, dict):
            # 检查是否是格式化后的输入（来自提示模板）
            if "context" in input and "question" in input:
                # 这是问答模板的输入
                context = input["context"]
                question = input["question"]
                messages = [
                    {"role": "system",
                     "content": f"你是一个有帮助的AI助手。请根据以下上下文回答问题。如果上下文不包含答案，请说“根据提供的上下文无法回答此问题”。\n\n上下文：{context}"},
                    {"role": "user", "content": question}
                ]
            elif "text" in input:
                # 这是总结模板的输入
                messages = [
                    {"role": "system", "content": "请对以下文本进行简洁准确的总结。"},
                    {"role": "user", "content": input["text"]}
                ]
            else:
                # 其他字典格式的输入
                messages = [
                    {"role": "user", "content": str(input)}
                ]
        else:
            # 其他格式的输入
            messages = [
                {"role": "user", "content": str(input)}
            ]

        result = self._call_api(messages, stream=False)
        return result['choices'][0]['message']['content']

    def stream(self, input, config=None, **kwargs):
        # 简化处理，实际streaming需要更复杂的处理
        return self.invoke(input, config, **kwargs)


class RAGSystem:
    """
    RAG系统主类
    """

    def __init__(self,
                 vllm_base_url: str = "http://localhost:23333",
                 embedding_model_name: str = "../models/BAAI/bge-small-zh-v1.5"):
        # 初始化本地VLLM模型
        self.chat_model = LocalVLLMChatModel(base_url=vllm_base_url)

        # 初始化嵌入模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},  # 可根据实际情况改为 'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )

        # 初始化向量数据库
        self.vectorstore = None
        self.retriever = None

        # 定义提示模板
        self.qa_template = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个有帮助的AI助手。请根据以下上下文回答问题。如果上下文不包含答案，请说“根据提供的上下文无法回答此问题”。\n\n上下文：{context}"),
            ("human", "{question}")
        ])

        self.summary_template = ChatPromptTemplate.from_messages([
            ("system", "请对以下文本进行简洁准确的总结。"),
            ("human", "{text}")
        ])

    def load_documents(self, file_path: str) -> List:
        """
        加载文档，支持 .txt, .pdf, .docx
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")

        documents = loader.load()
        return documents

    def process_documents(self, documents: List, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        处理文档（分割、嵌入、存储）
        """
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)

        # 创建向量数据库
        self.vectorstore = FAISS.from_documents(split_docs, self.embedding_model)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        print(f"成功处理 {len(split_docs)} 个文档块")

    def query(self, question: str) -> str:
        """
        对问题进行RAG查询
        """
        if not self.retriever:
            return "请先加载文档并处理"

        # 创建RAG链
        # 修改链的构建方式，因为我们现在直接处理格式化输入
        def format_context(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
                {"context": self.retriever | format_context, "question": RunnablePassthrough()}
                | self.chat_model
        )

        return rag_chain.invoke(question)

    def summarize_document(self) -> str:
        """
        对整个文档进行总结
        """
        if not self.vectorstore:
            return "请先加载文档并处理"

        # 获取所有文档块
        # all_docs = self.vectorstore.similarity_search("", k=min(30, len(self.vectorstore.docstore)))  # 限制最多30个块
        docstore_length = len(self.vectorstore.docstore._dict)
        all_docs = self.vectorstore.similarity_search("", k=min(30, docstore_length))

        def format_context(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        context = format_context(all_docs)

        # 构建专门用于文档总结的提示
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个专业的文档总结助手。请根据提供的完整文档内容，给出一个全面、准确的总结。总结应涵盖文档的主要观点和关键信息。"),
            ("human", "文档内容：{context}\n\n请总结以上文档的主要内容。")
        ])

        # 创建总结链
        summary_chain = (
                summary_prompt
                | self.chat_model
        )

        return summary_chain.invoke({"context": context})

    def add_document(self, file_path: str):
        """
        添加新文档到现有向量数据库
        """
        documents = self.load_documents(file_path)
        self.process_documents(documents)


# 示例使用
if __name__ == "__main__":
    # 创建RAG系统实例
    # 注意：请确保你的VLLM服务在 http://localhost:23333 运行，并且模型 Qwen3-4B-Instruct-2507 已加载
    rag_system = RAGSystem(
        vllm_base_url="http://localhost:23333",  # VLLM服务地址
        embedding_model_name="../models/BAAI/bge-small-zh-v1.5"  # 嵌入模型
    )

    # 加载并处理文档
    # 请将 "sample_document.txt" 替换为你的文档路径
    # 支持 .txt, .pdf, .docx 格式
    try:
        docs = rag_system.load_documents("../dataset/K系列调试作业指导书.pdf")  # 替换为你的文档路径
        rag_system.process_documents(docs)
        print("文档加载并处理完成")
    except Exception as e:
        print(f"加载文档时出错: {e}")
        print("使用示例问题跳过文档加载...")

    # 示例问题
    question = "简述水冷机安装的步骤 ？"
    if rag_system.retriever:
        answer = rag_system.query(question)
        print(f"问题: {question}")
        print(f"答案: {answer}")
    else:
        print("未加载文档，无法进行RAG查询")

    summary = rag_system.summarize_document()
    print(f"总结: {summary}")