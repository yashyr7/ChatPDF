import os
from pathlib import Path
from llama_index import VectorStoreIndex, ServiceContext, download_loader, set_global_service_context, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI

os.environ["OPENAI_API_KEY"] = 'API_KEY'

# define LLM
llm = OpenAI(model="gpt-4", temperature=0, max_tokens=256)

# configure service context
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context)


storage_context = StorageContext.from_defaults(persist_dir="./db/")
index = load_index_from_storage(storage_context)

if index is None:
    PDFMinerReader = download_loader("PDFMinerReader")
    loader = PDFMinerReader()
    documents = loader.load_data(file=Path('*YOUR PDF*'))
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir="./db/")

query= """YOUR QUERY"""

print(query + "\n-------------------------------\n")

query_engine = index.as_query_engine()

results = query_engine.query(query)


print(results)
