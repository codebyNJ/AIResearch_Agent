import streamlit as st
import os
import re
import requests
import json
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    HfApiModel,
    ManagedAgent,
    DuckDuckGoSearchTool,
    tool,
)
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Check for HF token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("HuggingFace token not found in .env file. Please add HF_TOKEN=your_token_here to your .env file.")
    st.stop()

# Log in to Hugging Face
login(HF_TOKEN)


@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage and converts its content to markdown format.

    Args:
        url: The complete URL of the webpage to visit (e.g., 'https://example.com').
            Must be a valid HTTP or HTTPS URL.

    Returns:
        str: The webpage content converted to Markdown format with the reference webpages links.
            Returns an error message if the request fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return markdown_content
    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


def format_agent_response(response):
    """Format the agent's response into a readable string. In a proper sentence that human being can understand with bullet points."""
    if isinstance(response, dict):
        formatted_parts = []
        
        if 'thoughts' in response:
            formatted_parts.append("## Thought Process")
            formatted_parts.append(response['thoughts'])
        
        if 'observations' in response:
            formatted_parts.append("## Observations")
            formatted_parts.append(response['observations'])
        
        if 'answer' in response:
            formatted_parts.append("## Answer")
            formatted_parts.append(response['answer'])
        
        if not formatted_parts:
            formatted_parts.append("## Results")
            for key, value in response.items():
                formatted_parts.append(f"### {key.title()}")
                formatted_parts.append(str(value))
        
        return "\n\n".join(formatted_parts)
    else:
        return str(response)


# Initialize the model and agents
@st.cache_resource
def initialize_agents():
    # Initialize the model
    model = HfApiModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=HF_TOKEN
    )
    web_agent = ToolCallingAgent(
        tools=[DuckDuckGoSearchTool(), visit_webpage],
        model=model,
        max_steps=10,
    )
    managed_web_agent = ManagedAgent(
        agent=web_agent,
        name="search",
        description="Runs web searches for you. Give it your query as an argument.",
    )
    manager_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[managed_web_agent],
        additional_authorized_imports=["time", "numpy", "pandas"],
    )
    return manager_agent


# Cache webpage content
@st.cache_data
def fetch_webpage_content(url):
    return visit_webpage(url)


# Sidebar
with st.sidebar:
    st.title("🔍 Search Settings")
    st.markdown("---")
    
    max_results = st.slider("Max Results", 1, 10, 5)
    search_depth = st.select_slider(
        "Search Depth",
        options=["Basic", "Moderate", "Deep"],
        value="Moderate"
    )
    
    st.markdown("---")
    st.markdown("### 📜 Search History")
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    for hist in st.session_state.search_history[-5:]:
        st.markdown(f"• {hist}")


# Main content
st.title("🔍 Web Research Assistant")
st.markdown("### Intelligent Web Search and Analysis")

# Initialize agents
manager_agent = initialize_agents()

# Search input
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("Enter your research query", placeholder="What would you like to know?")
with col2:
    search_button = st.button("🔎 Search", type="primary", disabled=not query)

if search_button and query:
    # Add to search history
    if query not in st.session_state.search_history:
        st.session_state.search_history.append(query)
    
    with st.spinner("🕵️‍♂️ Researching..."):
        try:
            # Create tabs for different views
            result_tab, sources_tab, analysis_tab = st.tabs(["📝 Results", "🔗 Sources", "📊 Analysis"])
            
            # Perform the search
            raw_result = manager_agent.run(query)
            
            # Format the result
            formatted_result = format_agent_response(raw_result)
            
            # Extract sources (URLs) from the formatted result
            sources = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', formatted_result)
            
            with result_tab:
                st.markdown("### 📊 Research Results")
                st.markdown(formatted_result)
                
                # Add export buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Results",
                        formatted_result,
                        file_name="research_results.md",
                        mime="text/markdown"
                    )
                with col2:
                    if isinstance(raw_result, dict):
                        st.download_button(
                            "📥 Download Raw JSON",
                            json.dumps(raw_result, indent=2),
                            file_name="research_results.json",
                            mime="application/json"
                        )
            
            with sources_tab:
                st.markdown("### 📚 Sources Referenced")
                if sources:
                    for idx, source in enumerate(sources, 1):
                        with st.expander(f"Source {idx}: {source}"):
                            # Add a loading spinner while fetching content
                            with st.spinner(f"Loading content from source {idx}..."):
                                # Fetch and display webpage content
                                content = fetch_webpage_content(source)
                                
                                # Create columns for metadata and controls
                                meta_col1, meta_col2 = st.columns([3, 1])
                                with meta_col1:
                                    st.markdown(f"**URL:** [{source}]({source})")
                                with meta_col2:
                                    st.download_button(
                                        "📥 Download Source",
                                        content,
                                        file_name=f"source_{idx}.md",
                                        mime="text/markdown"
                                    )
                                
                                st.markdown("---")
                                st.markdown("### Content Preview")
                                # Display a preview of the content with a character limit
                                preview_length = 1000
                                if len(content) > preview_length:
                                    st.markdown(content[:preview_length] + "...")
                                    st.markdown("*Content truncated. Download the full source using the button above.*")
                                else:
                                    st.markdown(content)
                else:
                    st.info("No external sources were referenced in this search.")
            
            with analysis_tab:
                st.markdown("### 📈 Content Analysis")
                
                # Create some metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sources Found", len(sources))
                with col2:
                    st.metric("Content Length", len(formatted_result))
                with col3:
                    st.metric("Sections", len(formatted_result.split("##")) - 1)
                
        except Exception as e:
            st.error(f"🚨 An error occurred: {str(e)}")
            st.markdown("""
            Please try:
            - Rephrasing your query
            - Checking your internet connection
            - Trying again in a few moments
            """)


# Footer with quick tips and stats
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 🚀 Quick Tips")
    st.markdown("""
    - Be specific in your queries
    - Use relevant keywords
    - Check multiple sources
    """)
with col2:
    st.markdown("### 🎯 Features")
    st.markdown("""
    - Intelligent search
    - Source verification
    - Export capabilities
    """)
with col3:
    st.markdown("### 📊 Stats")
    st.markdown(f"""
    - Searches today: {len(st.session_state.search_history)}
    - Sources analyzed: {len(sources) if 'sources' in locals() else 0}
    """)