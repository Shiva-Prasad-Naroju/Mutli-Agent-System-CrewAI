import os
import datetime
from dotenv import load_dotenv
import markdown
from langchain_groq.chat_models import ChatGroq
from crewai import Agent, Task, Crew

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inputs
topic = "Artificial Intelligence"
crew_inputs = {"topic": topic}

# Define Agents
planner = Agent(
    llm=ChatGroq(api_key=GROQ_API_KEY, temperature=0, model="llama-3.1-8b-instant"),
    role="Content Planner",
    goal=f"Plan engaging and factually accurate content on {topic}",
    backstory=f"You're planning a blog article about {topic}. "
              "Your output is an outline and key points that the Content Writer will use.",
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    llm=ChatGroq(api_key=GROQ_API_KEY, temperature=0, model="llama-3.1-8b-instant"),
    role="Content Writer",
    goal=f"Write insightful and factually accurate blog post about {topic}",
    backstory=f"You're writing a blog based on the Content Planner's outline for {topic}. "
              "Provide balanced insights backed by the plan, with sections, subtitles, and 2-3 paragraphs per section.",
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    llm=ChatGroq(api_key=GROQ_API_KEY, temperature=0, model="llama-3.1-8b-instant"),
    role="Editor",
    goal="Edit the blog post to ensure clarity, grammar, professional style, and readability",
    backstory="You are an editor who receives a blog post from the Writer. "
              "Your job is to refine grammar, structure sections, and polish style. "
              "Do not summarize or remove content, only enhance it.",
    allow_delegation=False,
    verbose=True
)

# Define Tasks
plan_task = Task(
    description=f"""
1. Prioritize the latest trends, key players, and noteworthy news on {topic}.
2. Identify the target audience and their interests.
3. Develop a detailed content outline including introduction, key points, and a call to action.
4. Include SEO keywords and relevant sources.
""",
    expected_output="A comprehensive content plan document (outline, key points, resources).",
    agent=planner,
)

write_task = Task(
    description=f"""
1. Use the content plan to craft a compelling blog post on {topic}.
2. Incorporate SEO keywords naturally.
3. Ensure sections/subtitles are engaging.
4. Structure with introduction, body, and conclusion.
5. Proofread for grammatical errors and style.
""",
    expected_output="A complete blog post in markdown format, ready for editing.",
    agent=writer,
)

edit_task = Task(
    description="Polish the blog post from the Writer for grammar, clarity, structure, and professional style.",
    expected_output="The full polished blog post in markdown format.",
    agent=editor
)

# Create Crew and run tasks
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan_task, write_task, edit_task],
    verbose=2
)

# Kickoff the Crew
final_result = crew.kickoff(inputs=crew_inputs)

# Save Markdown
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
topic_str = topic.replace(" ", "_")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
md_filename = f"{output_dir}/{topic_str}_{timestamp}.md"

with open(md_filename, "w", encoding="utf-8") as f:
    f.write(final_result)
print(f"Markdown output saved to: {md_filename}")

# Convert Markdown to Premium HTML
html_content = markdown.markdown(final_result)
html_filename = f"{output_dir}/{topic_str}_{timestamp}.html"

with open(html_filename, "w", encoding="utf-8") as f:
    f.write(f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{topic}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: "Georgia", serif; line-height: 1.7; background: #f5f5f5; color: #333; padding: 0 20px; }}
a {{ color: #1a73e8; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.container {{ max-width: 800px; margin: 50px auto; background: #fff; padding: 50px; box-shadow: 0 8px 24px rgba(0,0,0,0.1); border-radius: 8px; }}
h1 {{ font-size: 2.5em; margin-bottom: 20px; font-weight: bold; line-height: 1.2; }}
h2, h3 {{ margin-top: 30px; margin-bottom: 15px; font-weight: bold; color: #222; }}
p {{ margin-bottom: 20px; }}
ul {{ margin-bottom: 20px; padding-left: 20px; }}
li {{ margin-bottom: 10px; }}
blockquote {{ border-left: 4px solid #1a73e8; padding-left: 15px; color: #555; font-style: italic; margin: 20px 0; }}
code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 4px; font-family: monospace; }}
img {{ max-width: 100%; margin: 20px 0; border-radius: 6px; }}
.footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #777; font-size: 0.9em; }}
@media (max-width: 600px) {{ h1 {{ font-size: 2em; }} h2 {{ font-size: 1.5em; }} }}
</style>
</head>
<body>
<div class="container">
{html_content}
<div class="footer">&copy; 2025 | Crafted with AI /div>
</div>
</body>
</html>
""")
print(f"HTML output saved to: {html_filename}")
