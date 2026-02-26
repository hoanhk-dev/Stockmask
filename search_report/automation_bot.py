from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import asyncio
import json
from utils import save_graph_as_png, PlaywrightCrawler
import csv
import asyncio

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# =========================
# STATE
# =========================

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_url: str
    links: list
    visited_urls: list  # Track visited URLs to detect cycles

# =========================
# CONTROLLER
# =========================

class AutomationBot:

    def __init__(self, headless: bool = False, max_iterations: int = 5, report_type: str = "Integrated Report"):
        self.crawler = PlaywrightCrawler(headless=headless)
        self.llm = ChatOpenAI(model="gemini-2.5-flash")
        self.max_iterations = max_iterations
        self.report_type = report_type
        self.app = self.build_graph()

    # -------------------------
    # CRAWL NODE
    # -------------------------
    async def crawl_node(self, state: AgentState):
        url = state["current_url"]
        print(f"\nCrawling: {url}")

        links = await self.crawler.extract_links(url)
        print(f"🔗 Extracted {len(links)} links from {url}")

        return {
            "links": links
        }

    # -------------------------
    # LLM NODE
    # -------------------------
    async def llm_node(self, state: AgentState):

        prompt_text = f"""
        You are a web navigation expert.
        Below are the navigation links extracted from the current webpage.
        Extracted Links:
        {state['links']}
        Your task:
        1. Determine whether the FULL {self.report_type} PDF already exists in the provided links.
        - A full report PDF typically:
            - Ends with ".pdf"
            - Contains keywords such as:
            "{self.report_type.lower()}", "report", "ir_all"
        - Do NOT select partial PDFs (e.g., ir_p1-10.pdf, section-based files).
        - Latest report is preferred if multiple similar PDFs are found.

        2. If a FULL {self.report_type} PDF is found:
        Return:
        {{
            "next_step": "END",
            "url": "<full_pdf_url>"
        }}

        3. If a FULL PDF is NOT found:
        Select the single most relevant next page to visit.
        Then return:
        {{
            "next_step": "continue_find",
            "url": "<best_next_url>"
        }}
        IMPORTANT:
        - Return ONLY one JSON object.
        - Do NOT include explanations.
        - Do NOT include any text outside the JSON.
        """

        response = await self.llm.ainvoke(prompt_text)

        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()
        content_json = json.loads(content)
        print("🤖 LLM:", content_json)

        return {
            "messages": [AIMessage(content=json.dumps(content_json))],
            "current_url": content_json["url"]
        }

    # -------------------------
    # ROUTER
    # -------------------------
    def route(self, state: AgentState):
        last = state["messages"][-1].content
        iteration_count = len(state["messages"])

        try:
            response_json = json.loads(last) if isinstance(last, str) else last
            next_step = response_json.get("next_step", "END")
            next_url = response_json.get("url")
            
            # Check if max iterations reached
            if iteration_count >= self.max_iterations * 2:
                print(f"⚠️ Max iterations ({self.max_iterations}) reached. Stopping.")
                return "end"
            
            # Check for cycles (same URL visited multiple times)
            visited_urls = state.get("visited_urls", [])
            if next_url in visited_urls:
                print(f"🔄 Cycle detected! URL already visited: {next_url}")
                return "end"
            
            if next_step == "continue_find" and next_url:
                # Add current URL to visited list
                state["visited_urls"] = visited_urls + [state["current_url"]]
                state["current_url"] = next_url
                print(f"📍 Next URL: {next_url}")
                return "crawl"
            else:
                print(f"✅ Found PDF: {next_url}")
                return "end"
                
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse LLM response: {e}")
            return "end"

    # -------------------------
    # BUILD GRAPH
    # -------------------------
        
    def build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("crawl", self.crawl_node)
        graph.add_node("llm", self.llm_node)
        
        graph.set_entry_point("crawl")
        graph.add_edge("crawl", "llm")
        graph.add_conditional_edges(
            "llm",
            self.route,
            {
                "crawl": "crawl",
                "end": END,
            }
        )

        app = graph.compile()

        try:
            save_graph_as_png(app)
        except Exception as e:
            print(f"Failed to save graph visualization: {e}")
        return app
    
    # -------------------------
    # RUN METHOD
    # -------------------------
    async def run(self, start_url: str):
        
        initial_state = {
            "messages": [],
            "current_url": start_url,
            "links": [],
            "visited_urls": []
        }
    
        final_state = await self.app.ainvoke(initial_state)

        return final_state

async def run_bot(auto_bot, company_info, output_file):
    """
    Run auto_bot for each company site and save PDF URL results to CSV.

    Parameters:
        auto_bot: your autonomous web bot (must have async .run())
        company_info: dict like {
            stock_id: {
                "name": company_name,
                "site": url
            }
        }
        output_file: str, CSV filename
    """

    # Create CSV header
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["stock_id", "company_name", "pdf_url"])

    # Main loop
    for stock_id, item in company_info.items():
        print(f"\n🔎 Processing {item['name']} ({stock_id})")

        try:
            results = await auto_bot.run(item["site"])
            url_pdf = results.get("current_url", "")

            print("✅ Found:", url_pdf)

        except Exception as e:
            url_pdf = f"ERROR: {str(e)}"
            print("❌ Error:", e)

        # Append result
        with open(output_file, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([stock_id, item["name"], url_pdf])

        await asyncio.sleep(2)  # tránh rate limit / block

    print(f"\n🎉 Done. Results saved to {output_file}")
