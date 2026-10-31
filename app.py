
import os
import re
from typing import TypedDict

from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langserve import add_routes

# Check API key
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.3
)

# Input structure
class ResumeInput(TypedDict):
    resume: str
    job_role: str

# Graph state
class ResumeState(TypedDict):
    resume: str
    job_role: str
    report: dict

# Output section names
SECTION_NAMES = {
    "1": "Resume Summary",
    "2": "Key Skills Found",
    "3": "Strengths",
    "4": "Skills to Improve",
    "5": "Resume Improvement Suggestions",
    "6": "Interview Preparation Topics"
}

# Convert AI response into separate sections
def format_report(content):

    if isinstance(content, list):
        content = "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    content = str(content)

    pattern = r"(?m)^#{0,3}\s*(\d)\.\s*(.+?)\s*$"
    matches = list(re.finditer(pattern, content))

    report = {}

    for i, match in enumerate(matches):
        number = match.group(1)

        if number not in SECTION_NAMES:
            continue

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)

        section_text = content[start:end].strip()

        # Remove markdown formatting
        section_text = re.sub(r"\*\*(.*?)\*\*", r"\1", section_text)
        section_text = re.sub(r"(?m)^\s*[-*]\s*", "", section_text)

        # Make the output easy to read in Playground
        section_text = " • ".join(
            line.strip()
            for line in section_text.splitlines()
            if line.strip()
        )

        report[SECTION_NAMES[number]] = section_text

    # Fallback if headings are not generated correctly
    if not report:
        report["Full Analysis"] = content.strip()

    return report


# Resume analysis node
def analyze_resume(state: ResumeState):

    prompt = ChatPromptTemplate.from_template("""
You are a professional AI Resume Analyzer.

Target Job Role: {job_role}

Candidate Resume:
{resume}

Analyze the resume carefully.

Provide a report using EXACTLY these six headings:

1. Resume Summary
2. Key Skills Found
3. Strengths
4. Skills to Improve
5. Resume Improvement Suggestions
6. Interview Preparation Topics

Instructions:
- Use simple and professional English.
- Give clear and relevant points.
- Use bullet points where appropriate.
- Focus on skills related to the target job role.
- Do not invent qualifications or experience.
- Do not assume skills that are not mentioned in the resume.
- Give practical improvement suggestions.
- Keep the report concise and under 400 words.
- Do not add any introduction or conclusion.
""")

    response = (prompt | llm).invoke({
        "resume": state["resume"],
        "job_role": state["job_role"]
    })

    report = format_report(response.content)

    return {"report": report}


# Create LangGraph workflow
workflow = StateGraph(ResumeState)

workflow.add_node("analyze_resume", analyze_resume)

workflow.add_edge(START, "analyze_resume")
workflow.add_edge("analyze_resume", END)

graph = workflow.compile()


# Run the AI Resume Analyzer
def run_resume_analyzer(inputs: ResumeInput):

    result = graph.invoke({
        "resume": inputs["resume"],
        "job_role": inputs["job_role"],
        "report": {}
    })

    return result["report"]


# Runnable for LangServe
agent = RunnableLambda(run_resume_analyzer)


# FastAPI application
app = FastAPI(
    title="AI Resume Analyzer Agent",
    description="An AI-powered agent that analyzes resumes and provides job-specific feedback.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "AI Resume Analyzer Agent is running successfully!"
    }


# LangServe route
add_routes(
    app,
    agent,
    path="/agent"
)
