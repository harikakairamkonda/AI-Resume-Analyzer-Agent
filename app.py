import os
from typing import TypedDict

from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langserve import add_routes

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

class ResumeInput(TypedDict):
    resume: str
    job_role: str

class ResumeState(ResumeInput):
    analysis: str

def analyze_resume(state: ResumeState):
    prompt = ChatPromptTemplate.from_template("""
You are a friendly AI Resume Analyzer.

Target job role: {job_role}

Resume:
{resume}

Analyze the resume and provide a concise report under 400 words.

Use these headings:
1. Resume Summary
2. Key Skills Found
3. Strengths
4. Skills to Improve for This Role
5. Resume Improvement Suggestions
6. Interview Preparation Topics

Use simple English and bullet points.
Do not invent qualifications or experience.
Identify skills relevant to the target role.
""")

    response = (prompt | llm).invoke(state)
    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    return {"analysis": content}

workflow = StateGraph(ResumeState)
workflow.add_node("analyze_resume", analyze_resume)
workflow.add_edge(START, "analyze_resume")
workflow.add_edge("analyze_resume", END)

graph = workflow.compile()

def run_resume_analyzer(inputs: ResumeInput):
    result = graph.invoke({
        "resume": inputs["resume"],
        "job_role": inputs["job_role"],
        "analysis": ""
    })
    return {"analysis": result["analysis"]}

agent = RunnableLambda(run_resume_analyzer)

app = FastAPI(
    title="AI Resume Analyzer Agent",
    description="Analyzes resumes and suggests improvements for a target job role.",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "AI Resume Analyzer Agent is running!"}

add_routes(app, agent, path="/agent")
