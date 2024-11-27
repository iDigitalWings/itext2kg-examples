import os

from langchain_community.chat_models import ChatZhipuAI
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field
from itext2kg import iText2KG
from itext2kg import DocumentsDistiller


class Character(BaseModel):
    name: str = Field(
        ..., description="人物名称"
    )
    description: str = Field(..., description="人物的介绍")

llm = ChatZhipuAI(model="glm-4-flash", api_key=os.environ.get("ZHIPU_API_KEY"))
embedding = OllamaEmbeddings(
    model="nomic-embed-text",
)

# Initialize the DocumentDistiller with llm model.
document_distiller = DocumentsDistiller(llm_model = llm)

# List of documents to be distilled.

base_path = os.path.abspath("../datasets/sanguo/")

documents = [os.path.join(base_path, f) for f in os.listdir(base_path)]

# Information extraction query.
IE_query = '''
# 指令：
- 您将阅读小说《三国演义》，
- 提取小说中的人物角色信息和关系，
- 使用中文。
'''

# Distill the documents using the defined query and output data structure.
distilled_doc = document_distiller.distill(documents=documents, IE_query=IE_query, output_data_structure=Character)

###############

# Initialize iText2KG with the llm model and embeddings model.
itext2kg = iText2KG(llm_model = llm, embeddings_model = embedding)

# Format the distilled document into semantic sections.
semantic_blocks = [f"{key} - {value}".replace("{", "[").replace("}", "]") for key, value in distilled_doc.items()]

# Build the knowledge graph using the semantic sections.
kg = itext2kg.build_graph(sections=semantic_blocks)

print('results >>>>>>')
print(kg)
for n in kg.entities:
    print(n.name)
for r in kg.relationships:
    print(f'{r.startEntity.name} ---> {r.endEntity.name}')

