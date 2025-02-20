import os
from langchain_openai import ChatOpenAI

api_key = os.environ["OPENAI_API_KEY"] 

llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)