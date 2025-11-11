# deploy_bge_m3_openai.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Optional
from modelscope import AutoModel, AutoTokenizer
import torch
from FlagEmbedding import BGEM3FlagModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


app = FastAPI(title="BGE-M3 Embedding Service - OpenAI Compatible")

# 加载模型
model_name = "./models/BAAI/bge-m3"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModel.from_pretrained(
#     model_name,
#     trust_remote_code=True,
#     device_map="auto",
#     torch_dtype=torch.float16
# )
model = BGEM3FlagModel(
     model_name,
     use_fp16=True
)


class EmbeddingRequest(BaseModel):
    model: str = "bge-m3"
    input: Union[str, List[str]]
    encoding_format: str = "float"
    user: Optional[str] = None

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    # 标准化输入
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    try:
        # 使用 FlagEmbedding 的 encode 方法
        outputs = model.encode(
            texts,
            batch_size=len(texts),
            max_length=8192,
            return_dense=True,
            return_sparse=False,      # LightRAG 只需要 dense
            return_colbert_vecs=False
        )

        dense_embeddings = outputs['dense_vecs']  # numpy array, shape (N, 1024)

        embedding_data = []
        for i, vec in enumerate(dense_embeddings):
            embedding_data.append(EmbeddingData(
                embedding=vec.tolist(),
                index=i
            ))

        # 简单估算 token 数（可选更精确的 tokenizer）
        total_tokens = sum(len(text.split()) for text in texts)  # 粗略估计

        return EmbeddingResponse(
            data=embedding_data,
            model=request.model,
            usage=Usage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )

    except Exception as e:
        import traceback
        print("Error:", str(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "bge-m3",
                "object": "model",
                "owned_by": "owner"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=23334)
