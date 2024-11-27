import os

from langchain_community.chat_models import ChatZhipuAI, ChatVertexAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PythonLoader


from itext2kg import iText2KG
from itext2kg import DocumentsDisiller


class Character(BaseModel):
    name: str = Field(
        ..., description="The name of the character， or the name of the place"
    )
    description: str = Field(..., description="The information of the charactor")


llm = ChatZhipuAI(model="glm-4-flash", api_key=os.environ.get("ZHIPU_API_KEY"))

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

itext2kg = iText2KG(llm_model=llm, embeddings_model=embeddings)
# semantic_blocks = [f"{key} - {value}".replace("{", "[").replace("}", "]") for key, value in distilled_doc.items()]


document_distiller = DocumentsDisiller(llm_model=llm)
IE_query = """
# 指令：
- 您有大量的小说故事阅读经验。
- 您将阅读小说《三国演义》，
- 您将提取小说中的人物角色信息和关系。
- 如果找不到正确的信息，请将其保留为空白。
"""

base_path = "../datasets/sanguo/"

texts = [os.path.join(base_path, f) for f in os.listdir(base_path)]


def build_sections(file_path):
    loader = PythonLoader(file_path)
    pages = loader.load_and_split()

    distilled_cv = document_distiller.distill(
        documents=[
            page.page_content.replace("{", "[").replace("}", "]") for page in pages
        ],
        IE_query=IE_query,
        output_data_structure=Character,
    )

    sections = [
        f"{key} - {value}".replace("{", "[").replace("}", "]")
        for key, value in distilled_cv.items()
        if value != [] and value != "" and value != None
    ]
    return sections


ent = []
rel = []


def build_graph(
    sections,
    existing_global_entities=None,
    existing_global_relationships=None,
    ent_threshold=2,
    rel_threshold=2,
):
    global_ent, global_rel = itext2kg.build_graph(
        sections=sections,
        ent_threshold=ent_threshold,
        rel_threshold=rel_threshold,
        existing_global_relationships=existing_global_relationships,
        existing_global_entities=existing_global_entities,
    )
    global ent, rel
    ent += global_ent
    rel += global_rel

    return global_ent, global_rel


for text in texts:
    build_graph(
        build_sections(text),
        existing_global_entities=ent,
        existing_global_relationships=rel,
        ent_threshold=1,
        rel_threshold=1,
    )


from pyvis.network import Network

net = Network(height="100vh", width="100%", bgcolor="#222222", font_color="#FFFFFF")
labels = [x["name"] for x in ent]
labels = list(set(labels))
for x in labels:
    net.add_node(x, color="#d48806", shape="box")
for x in rel:
    try:
        net.add_edge(x["startNode"], x["endNode"], weight=1)
    except Exception as e:
        print(e)
net.show("mygraph.html", notebook=False)
