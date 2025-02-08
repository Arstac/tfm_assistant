import pandas as pd
import os
from langchain_openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


df = pd.read_csv('../proyectos_construccion_dataset.csv', delimiter=';')

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    allow_dangerous_code=True
)


agent_tool = agent.as_tool(
    name="csv_agent",
    description="Agent for csv file",
)

agent.invoke("Plot hist of Presupuesto")