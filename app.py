import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq.chat_models import ChatGroq
from crewai import Agent, Task, Crew
import datetime
import markdown

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inputs for the Crew (dynamic topic)
crew_inputs = {"topic": "Artificial Intelligence"}
topic = crew_inputs["topic"]  # dynamic topic

# Planner Agent
planner = Agent(
    llm=ChatGroq(api_key=GROQ_API_KEY, temperature=0, model="llama-3.1-8b-instant"),
    role="Content Planner",
    goal=f"Plan engaging and factually accurate content on {topic}",
    backstory=f"You're working on planning a blog article about the topic: {topic}. "
              "You collect information that helps the audience learn something "
              "and make informed decisions. "
              "Your work is the basis for the Content Writer to write an article on this topic.",
    allow_delegation=False,
    verbose=True
)

# Writer Agent
writer = Agent(
    llm=ChatGroq(api_key=GROQ_API_KEY, temperature=0, model="llama-3.1-8b-instant"),
    role="Content Writer",
    goal=f"Write insightful and factually accurate opinion piece about the topic: {topic}",
    backstory=f"You're working on writing a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of the Content Planner, who provides an outline "
              "and relevant context about the topic. "
              "You follow the main objectives and direction of the outline, "
              "as provided by the Content Planner. "
              "You also provide objective and impartial insights "
              "and back them up with information provided by the Content Planner. "
              "You acknowledge in your opinion piece "
              "when your statements are opinions as opposed to objective statements.",
    allow_delegation=False,
    verbose=True
)

# Editor Agent
editor = Agent(
    llm=ChatGroq(api_key=GROQ_API_KEY, temperature=0, model="llama-3.1-8b-instant"),
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization.",
    backstory="You are an editor who receives a blog post from the Content Writer. "
              "Your goal is to review the blog post to ensure that it follows journalistic best practices, "
              "provides balanced viewpoints when providing opinions or assertions, "
              "and also avoids major controversial topics or opinions when possible.",
    allow_delegation=False,
    verbose=True
)

# Plan Task
plan = Task(
    description=f"""
1. Prioritize the latest trends, key players, and noteworthy news on {topic}.
2. Identify the target audience, considering their interests and pain points.
3. Develop a detailed content outline including an introduction, key points, and a call to action.
4. Include SEO keywords and relevant data or sources.
""",
    expected_output="A comprehensive content plan document with an outline, audience analysis, SEO keywords, and resources.",
    agent=planner,
)

# Write Task
write = Task(
    description=f"""
1. Use the content plan to craft a compelling blog post on {topic}.
2. Incorporate SEO keywords naturally.
3. Sections/Subtitles are properly named in an engaging manner.
4. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.
5. Proofread for grammatical errors and alignment with the brand's voice.
""",
    expected_output="A well-written blog post in markdown format, ready for publication, each section should have 2 or 3 paragraphs.",
    agent=writer,
)

# Edit Task
edit = Task(
    description="Proofread the given blog post for grammatical errors and alignment with the brand's voice.",
    expected_output="A well-written blog post in markdown format, ready for publication, each section should have 2 or 3 paragraphs.",
    agent=editor
)

# Creating the crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=2
)

# Running the crew
result = crew.kickoff(inputs=crew_inputs)
print(result)

# Saving the output as Markdown
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
topic_str = topic.replace(" ", "_")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
md_filename = f"{output_dir}/{topic_str}_{timestamp}.md"

with open(md_filename, "w", encoding="utf-8") as f:
    f.write(result)
print(f"Markdown output saved to: {md_filename}")

# Convert Markdown to HTML automatically
html_content = markdown.markdown(result)
html_filename = f"{output_dir}/{topic_str}_{timestamp}.html"

with open(html_filename, "w", encoding="utf-8") as f:
    f.write(f"""
<html>
<head>
    <title>{topic}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown-light.min.css">
    <style>
        body {{
            max-width: 900px;
            margin: auto;
            padding: 20px;
        }}
    </style>
</head>
<body class="markdown-body">
{html_content}
</body>
</html>
""")
print(f"HTML output saved to: {html_filename}")
