import os
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
import re
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import SystemMessage
from langsmith.run_helpers import traceable
from fastapi import FastAPI

load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")
serp_api_key = os.getenv("SERP_API_KEY")

@tool("search")
def search(query: str) -> str:
    """
    Google search to get latest information
    """
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serp_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    url: str = Field(description="The url of the website to scrape")
    objective: str = Field(description="The objective of the research")


@tool("scrape_website", args_schema=ScrapeWebsiteInput)
def scrape_website(url: str, objective: str) -> str:
    """
    scrape website, and also will summarize the content base on objective if content is too large
    objective is the original objective & task that user give to the agent, url is the website to be scraped
    """
    print("Scraping website...")

    # Send a GET request to the URL
    response = requests.get(url)

    # Check the response status code
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        get_text = soup.get_text()
        # Clean the text by removing extra spaces, carriage returns, and line feeds
        text = re.sub(r'\s+', ' ', get_text).strip()
        #print("CONTENT:", text)

        # Here you'd need a function `summary` defined elsewhere to summarize the content
        # Assuming 'summary' is a function that takes the 'objective' and the 'text',
        # and returns a summarized version of 'text'.
        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")

def summary(objective, content):    
    content = content[:16385]
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])

    prompt_template = """
    WEBSITE CONTENT: 
    "{text}"
    SUMMARY:
    """ + f"""
    --
    Above is the scrapped website content, please remove noise & filter out key content that help on research object: {objective};
    The summary should be detailed, with lots of reference & links to back up the research, as well as additional inofrmation for providing context
    EXTRACTED KEY CONTENT (Be detailed):
    """

    prompt = PromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")
    llm_chain = LLMChain(llm=llm,prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    summary = stuff_chain.run(docs)
    return summary

# class ScrapeWebsiteTool(BaseTool):
#     name = "scrape_website"
#     description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
#     args_schema: Type[BaseModel] = ScrapeWebsiteInput

#     def _run(self, objective: str, url: str):
#         return scrape_website(objective, url)

#     def _arun(self, url: str):
#         raise NotImplementedError("error here")
    
# 3. Create langchain agent with the tools above
tools = [scrape_website,search]


message = """You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")
memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="chat_history",
    max_token_limit=3000,
    return_messages=True)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            message,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent,tools=tools,memory=memory)

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/")
def researcherAgent(query: Query):
    query = query.query
    content = agent_executor.invoke({"input": query})
    act_content = content['output']
    return act_content

# import streamlit as st
# #4. Use streamlit to create a web app

# @traceable(run_type="chain")
# def main():
#     st.set_page_config(page_title="AI research agent", page_icon=":bird:")

#     st.header("AI research agent :bird:")
#     query = st.text_input("Research goal")

#     if query:
#         st.write("Doing research for ", query)

#         result = agent_executor.invoke({"input": query})

#         st.info(result['output'])


# if __name__ == '__main__':
#     main()

