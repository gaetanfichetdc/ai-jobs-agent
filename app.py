import streamlit as st
import os
import pandas as pd
import plotly.express as px
import uuid
import requests
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# ==========================================
# 1. Page Configuration & Custom UI Styling
# ==========================================
st.set_page_config(page_title="AI Jobs Data Agent", layout="wide", page_icon="🤖")

st.markdown("""
<style>
    [data-testid="stSidebar"] div[data-testid="stMetric"] {
        background-color: #1e293b !important; 
        padding: 15px !important; 
        border-radius: 10px !important;
        border: 1px solid #334155 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

MAX_CALLS_PER_USER = 15

@st.cache_resource
def get_global_rate_limiter():
    return {}

global_limiter = get_global_rate_limiter()

def get_user_identifier():
    if "fallback_uuid" not in st.session_state:
        st.session_state.fallback_uuid = str(uuid.uuid4())
    return st.session_state.fallback_uuid

user_key = get_user_identifier()
if user_key not in global_limiter:
    global_limiter[user_key] = 0

# ==========================================
# 2. Dataset Load & Sidebar Stats
# ==========================================
@st.cache_data
def load_data(): 
    # Clean string conversions on load to prevent type casting failures later
    data = pd.read_csv('ai_jobs_global_geocoded.csv')
    if 'lat' in data.columns:
        data['lat'] = pd.to_numeric(data['lat'], errors='coerce')
    if 'lon' in data.columns:
        data['lon'] = pd.to_numeric(data['lon'], errors='coerce')
    return data

df = load_data()

with st.sidebar:
    st.header("📊 Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Total Jobs", len(df))
    
    unique_cities = df['city'].dropna().nunique() if 'city' in df.columns else df['country'].dropna().nunique()
    col2.metric("Unique Cities", unique_cities)
    st.markdown("---")
    remaining_queries = max(0, MAX_CALLS_PER_USER - global_limiter[user_key])
    st.info(f"🔑 Queries Remaining: {remaining_queries} / {MAX_CALLS_PER_USER}")

# ==========================================
# 3. Helpers: Safe Multi-Target Location Filter
# ==========================================
def filter_dataframe_by_location(data: pd.DataFrame, location_query: str | None) -> pd.DataFrame:
    if not location_query or location_query.lower() == "all":
        return data
    
    tokens = [t.strip().lower() for t in location_query.split(",") if t.strip()]
    masks = []
    
    for q in tokens:
        # Robust substring matching to bypass strict equality limits
        if q in ["usa", "united states", "us", "u.s.", "u.s.a."]:
            pattern = "us|united states|america"
            c_mask = data['country'].str.lower().str.contains(pattern, na=False) if 'country' in data.columns else pd.Series(False, index=data.index)
            t_mask = data['city'].str.lower().str.contains(pattern, na=False) if 'city' in data.columns else pd.Series(False, index=data.index)
            mask = c_mask | t_mask
        elif q in ["uk", "united kingdom", "gbr"]:
            pattern = "uk|united kingdom|great britain|england"
            c_mask = data['country'].str.lower().str.contains(pattern, na=False) if 'country' in data.columns else pd.Series(False, index=data.index)
            t_mask = data['city'].str.lower().str.contains(pattern, na=False) if 'city' in data.columns else pd.Series(False, index=data.index)
            mask = c_mask | t_mask
        else:
            c_mask = data['country'].str.lower().str.contains(q, na=False) if 'country' in data.columns else pd.Series(False, index=data.index)
            t_mask = data['city'].str.lower().str.contains(q, na=False) if 'city' in data.columns else pd.Series(False, index=data.index)
            mask = c_mask | t_mask
        masks.append(mask)
    
    if not masks:
        return data
    return data[pd.concat(masks, axis=1).any(axis=1)]

# ==========================================
# 4. Optimized Plotly Chart Engine
# ==========================================
def render_plotly_chart(chart_type: str, column: str, group_by: str | None = None, countries_filter: str | None = "All", key: str | None = None):
    plot_df = filter_dataframe_by_location(df.copy(), countries_filter)
    
    if plot_df.empty:
        st.warning(f"⚠️ No rows matched the filter condition: '{countries_filter}'")
        return

    if chart_type == "map":
        with st.spinner("Generating city scatter map..."):
            if 'lat' not in plot_df.columns or 'lon' not in plot_df.columns:
                st.error("❌ Spatial Error: 'lat' or 'lon' coordinate columns are missing from the dataset.")
                return
                
            # Clear invalid/blank coordinate properties safely
            map_df = plot_df.dropna(subset=['lat', 'lon']).copy()
            
            if map_df.empty:
                st.error(f"❌ Geographic Failure: Matched records for '{countries_filter}', but those rows contain blank coordinates.")
                return
                
            if 'full_location' not in map_df.columns:
                if 'city' in map_df.columns:
                    map_df['full_location'] = map_df['city'].astype(str) + ", " + map_df['country'].astype(str)
                else:
                    map_df['full_location'] = map_df['country'].astype(str)
            
            # Group by geographic signatures directly
            map_data = map_df.groupby(['lat', 'lon', 'full_location']).size().reset_index(name='Job Offers')
                
            fig = px.scatter_geo(
                map_data, lat='lat', lon='lon', size='Job Offers', color='Job Offers',
                hover_name='full_location', size_max=22,
                title=f"City-Level Distribution of AI Job Offers ({countries_filter.upper()})",
                color_continuous_scale=px.colors.sequential.Plasma
            )
            
            q_clean = countries_filter.lower().strip()
            if any(term in q_clean for term in ["usa", "united states", "us"]):
                fig.update_geos(scope="usa", showlakes=True, lakecolor="rgb(255, 255, 255)")
            elif any(term in q_clean for term in ["europe", "germany", "uk", "united kingdom"]):
                fig.update_geos(scope="europe")
            else:
                fig.update_geos(projection_type="natural earth")
                
            fig.update_layout(
                margin={"r":0,"t":50,"l":0,"b":0}, 
                geo=dict(bgcolor='rgba(0,0,0,0)', showland=True, landcolor="rgb(243, 244, 246)")
            )

    elif chart_type == "box":
        target_col = 'salary_max' if column == 'salary' else column
        fig = px.box(plot_df, y=target_col, x=group_by, color=group_by, title="Salary Distributions")
        
    else: 
        active_color = group_by if group_by in plot_df.columns else None
        if not active_color and (',' in str(countries_filter) or countries_filter.lower() == 'all'):
            active_color = 'country'
            
        target_col = 'salary_max' if column == 'salary' else column
        fig = px.histogram(
            plot_df, x=target_col, color=active_color,
            barmode='group', opacity=0.75, title="Salary Distribution Comparison"
        )
        fig.update_layout(bargap=0.15)

    st.plotly_chart(fig, use_container_width=True, key=key)

# ==========================================
# 5. Agent Tools
# ==========================================
@tool
def calculate_job_stat(metric: str, column: str, country: str = "All"):
    """Calculates statistical operations (mean, max, min, count) for filtered segments."""
    data = filter_dataframe_by_location(df, country)
    if data.empty: return f"0 rows found for location filter context: '{country}'."
    if metric.lower() == "count": return f"{len(data)} total positions found."
    
    target = 'salary_max' if column == 'salary' else column
    if metric == "mean": return f"${float(data[target].mean()):,.2f}"
    if metric == "max": return f"${float(data[target].max()):,.2f}"
    return f"${float(data[target].min()):,.2f}"

@tool
def plot_job_data(chart_type: str, column: str, group_by: str | None = None, countries_filter: str | None = "All"):
    """Generates high-resolution data visualizations ('histogram', 'box', or 'map')."""
    check_df = filter_dataframe_by_location(df.copy(), countries_filter)
    if check_df.empty: return f"Error: No records match '{countries_filter}'."
    return f"Successfully initialized the layout for {chart_type}."

# ==========================================
# 6. Agent Orchestration Loop
# ==========================================
system_prompt = f"""
You are an AI Data Analyst agent with a fully loaded dataset of {len(df)} AI job postings across {df['country'].nunique()} countries.
The data is already in memory — never suggest sample code, never ask the user for data, never show code snippets.
When a tool call succeeds, give a short natural-language summary of the result (1-3 sentences). Do not repeat tool arguments or show implementation details.

Tool usage:
- For maps: chart_type="map", column="country", countries_filter=<location>.
- For salary comparisons: chart_type="histogram", column="salary", countries_filter=<locations comma-separated>, group_by="country".
- For salary distributions: chart_type="box", column="salary", group_by=<grouping column>.
- For statistics: use calculate_job_stat with metric (mean/max/min/count), column, and country filter.

Available columns: {', '.join(df.columns.tolist())}
"""

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
llm_with_tools = llm.bind_tools([calculate_job_stat, plot_job_data])

if "messages" not in st.session_state: 
    st.session_state.messages = []

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])
        if "chart_args" in msg and msg["chart_args"]:
            render_plotly_chart(**msg["chart_args"], key=f"hist_{idx}")

if global_limiter[user_key] < MAX_CALLS_PER_USER:
    if prompt := st.chat_input(placeholder="Ask me anything about the data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)

        with st.chat_message("assistant"):
            global_limiter[user_key] += 1
            langchain_messages = [SystemMessage(content=system_prompt)]
            for m in st.session_state.messages:
                role = HumanMessage if m["role"] == "user" else AIMessage
                langchain_messages.append(role(content=m["content"]))

            response = llm_with_tools.invoke(langchain_messages)
            chart_to_save = None
            
            if response.tool_calls:
                tc = response.tool_calls[0]
                if tc['name'] == 'calculate_job_stat':
                    tool_output = calculate_job_stat.invoke(tc['args'])
                else:
                    tool_output = plot_job_data.invoke(tc['args'])
                    if "Error" not in str(tool_output): 
                        chart_to_save = tc['args']
                
                langchain_messages.append(response)
                langchain_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tc['id']))
                final_response = llm.invoke(langchain_messages)
                assistant_content = final_response.content
                st.markdown(assistant_content)
                if chart_to_save:
                    render_plotly_chart(**chart_to_save, key="active_chart")
            else:
                assistant_content = llm.invoke(langchain_messages).content
                st.markdown(assistant_content)
                
            st.session_state.messages.append({
                "role": "assistant", 
                "content": assistant_content, 
                "chart_args": chart_to_save
            })
            st.rerun()