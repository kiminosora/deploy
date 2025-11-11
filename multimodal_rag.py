import arxiv
import os
import getpass
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from IPython.display import HTML, display
from PIL import Image
import base64
from paddlex import create_model
from pdf2image import convert_from_path
import cv2
import numpy as np
from dotenv import load_dotenv
import json
import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS


# 图像编码
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def layoutdectect(image_path):
    model_name = "PP-DocLayout-S"
    output_dir = "./localrag/rag2/output/"
    model = create_model(model_name=model_name)
    output = model.predict(image_path, batch_size=1, layout_nms=True)
    file_name = image_path.split("/")[-1].split('.')[0]
    # print("file_name : ",file_name)
    for res in output:
        res.print()
        res.save_to_img(save_path=f"{output_dir}/{file_name}_label.png")
        res.save_to_json(save_path=f"{output_dir}/{file_name}_label.json")
    return f"{output_dir}/{file_name}_label.png", f"{output_dir}/{file_name}_label.json"


def extract_image_and_table(pdf_path, output_dir, extract_path):
    # 获取文件名称，不包含路径和后级
    file_li = pdf_path.split('/')[-1].split('.')
    file_name = pdf_path.split('/')[-1].split('.')[0]
    if len(file_li) > 2:
        file_name = file_name + "." + pdf_path.split('/')[-1].split('.')[1]
    images = convert_from_path(pdf_path)
    print(f"Number of images: {len(images)}")
    all_images_path = []
    for i, image in enumerate(images):
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        Image.fromarray(image_np).save(f'{output_dir}{file_name}_{i}.png', 'PNG')
        # layout_result =model.detect(image_np)
        _, res_detect_path = layoutdectect(f'{output_dir}{file_name}_{i}.png')
        with open(res_detect_path, 'r') as f:
            layout_result = json.load(f)
        box = layout_result["boxes"]
        blocks = [b for b in box if b["label"] == "image"]
        for j, block in enumerate(blocks):
            coordinate = block["coordinate"]
            x1 = int(coordinate[0])
            y1 = int(coordinate[1])
            x2 = int(coordinate[2])
            y2 = int(coordinate[3])
            cropped_image = Image.fromarray(image_np).crop((x1, y1, x2, y2))
            cropped_image.save(f'{extract_path}{file_name}_image_{i}_{j}.png', 'PNG')
            all_images_path.append(f'{extract_path}{file_name}_image_{i}_{j}.png')
        blocks = [b for b in box if b["label"] == "chart"]
        for j, block in enumerate(blocks):
            coordinate = block["coordinate"]
            x1 = int(coordinate[0])
            y1 = int(coordinate[1])
            x2 = int(coordinate[2])
            y2 = int(coordinate[3])
            cropped_image = Image.fromarray(image_np).crop((x1, y1, x2, y2))
            cropped_image.save(f'{extract_path}{file_name}_chart_{i}_{j}.png', 'PNG')
            all_images_path.append(f'{extract_path}{file_name}_chart_{i}_{j}.png')
        blocks = [b for b in box if b["label"] == "table"]
        for j, block in enumerate(blocks):
            coordinate = block["coordinate"]
            x1 = int(coordinate[0])
            y1 = int(coordinate[1])
            x2 = int(coordinate[2])
            y2 = int(coordinate[3])
            cropped_image = Image.fromarray(image_np).crop((x1, y1, x2, y2))
            cropped_image.save(f'{extract_path}{file_name}_table_{i}_{j}.png', 'PNG')
            all_images_path.append(f'{extract_path}{file_name}_table_{i}_{j}.png')
    print("all images path : ", all_images_path)
    return all_images_path


pdf_path = "./dataset/K系列调试作业指导书.pdf"
output_dir = "./localrag/rag2/output/"
extract_path = "./localrag/rag2/extract/"
all_images_path = extract_image_and_table(pdf_path, output_dir, extract_path)

client = OpenAI(api_key='YOUR_API_KEY', base_url='http://0.0.0.0:23333/v1')


def chatwithimage(img_base, prompt):
    model_name = client.models.list().data[0].id
    content = []
    # 添加所有图片（转换为 base64 data URL）
    for img_path in img_base:
        content.append({
            "type": "image_url",
            "image_url": {"url": img_path}
        })

    # 添加文本 prompt
    content.append({
        "type": "text",
        "text": prompt
    })
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
    )
    return response.choices[0].message.content


def generate_img_summaries(image_path_list):
    img_base64_list = []
    image_summaries = []
    prompt = "你是一位负责为检索图像生成图像总结的助手。这些总结将被嵌入并用于检索原始图像。提供一个全面且不包含与原始图像无关内容的图像总结。"
    for i, img_file in enumerate(image_path_list):
        if img_file.endswith(".png"):
            # img_path = os.path.join(path,img_file)
            img_path = []
            img_path.append(img_file)
            base64_image = encode_image(img_file)
            print(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(chatwithimage(img_path, prompt))
    return img_base64_list, image_summaries


extract_images_path = all_images_path[:]
img_base64_list, image_summaries = generate_img_summaries(extract_images_path)
for i in image_summaries:
    print("\n", i)

em_model = "./models/BAAI/bge-large-zh-v1.5"


class BGEMilvusEmbeddings(Embeddings):
    def __init__(self):
        self.model = BGEM3EmbeddingFunction(
            model_name=em_model,  # Specify the model name
            device='cpu',  # Specify the device to use, e.g.,'cpu'or 'cuda:0"
            use_fp16=False  # Specify whether to use fpl6. Set to 'False' if 'device`is 'cpu'.
        )

    def embed_documents(self, texts):
        embeddings = self.model.encode_documents(texts)
        return [i.tolist() for i in embeddings["dense"]]

    def embed_query(self, text):
        embedding = self.model.encode_queries([text])
        return embedding["dense"][0].tolist()


embedding_model = BGEMilvusEmbeddings()


def saveimage2path(base64_content, index):
    # 将Base64字符串解码为二进制数据
    binary_data = base64.b64decode(base64_content)
    # 指定要保存的文件路径
    file_path = f"./localrag/rag2/result/{index}.png"  # 很据实际内容类型修改文件扩展名
    # 将二进制数据写入文件
    with open(file_path, "wb") as file:
        file.write(binary_data)
    print(f"文件已保存到{file_path}")
    return file_path


documents = [Document(page_content=image_summaries[i], metadata={"source": saveimage2path(img_base64_list[i], i)}) for i
             in range(len(image_summaries))]

# Create the Milvus vector store

vectorstore = FAISS.from_documents(documents, embedding_model)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 连接单机版
# vectorstore =Milvus.from_documents(
# documents,
# embedding_model,
# connection_args={"host": "127.0.0.1","port":"19530"},
# collection_name="multimodal_rag_demo"
# )
# # Create a retriever from the vector store
# retriever =vectorstore.as_retriever()

query = "简述激光头安装的步骤"
retrieved_docs = retriever.invoke(query, limit=3)
print("\n\n\n", len(retrieved_docs))
# plt_img_base64(retrieved_docs[e].metadata["source"])
res = []
for i in range(3):  # 假设我们只需要前三个元素
    if i < len(retrieved_docs):  # 确保索引不超出列表长度
        res.append(retrieved_docs[i].metadata["source"])
    else:
        break  # 如果列表长度小于3，则结束循环
# res = retrieved_docs[0].metadata["source"]
print(res)
# print("\n",retrieved_docs[0].page_content)
# base64_image=encode_image(res)
# prompt=f"你将获得一张图片或一个表格。利用图片或表格中的信息，提供与用户问题相关的具体答案。用户的问题：{query}"
prompt = f"根据图片回答问题，请不要自行编造，问题：{query}"
result = chatwithimage(res, prompt)
print(result)