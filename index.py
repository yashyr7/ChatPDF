import os
from pathlib import Path
from llama_index import VectorStoreIndex, StorageContext, download_loader, ServiceContext, set_global_service_context
from llama_index.llms import OpenAI

os.environ["OPENAI_API_KEY"] = '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'

# define LLM
llm = OpenAI(model="gpt-4", temperature=0, max_tokens=256)

# configure service context
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context)

PDFMinerReader = download_loader("PDFMinerReader")
loader = PDFMinerReader()
documents = loader.load_data(file=Path('./termtest1.pdf'))
index = VectorStoreIndex.from_documents(documents, show_progress=True)
index.storage_context.persist(persist_dir="./db/")

print("Indexing Done!!")
