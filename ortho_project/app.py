import streamlit as st
import os
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from anthropic import Anthropic
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain, LLMChain
from langchain.chat_models import ChatAnthropic
from langchain.prompts import PromptTemplate
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from neo4j import GraphDatabase
from graph_scheme import get_graph_schema
from dotenv import load_dotenv
import sqlite3
import base64
import logging
import re
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from typing import List
from datetime import datetime
from fuzzywuzzy import process
import Levenshtein 

# Clear all Streamlit cache
st.cache_data.clear()
st.cache_resource.clear()

# Global styles for the main app
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: white !important;
    }}
    .stApp .main {{
        background-color: white !important;
    }}
    .stApp .block-container {{
        background-color: white !important;
    }}
    .big-font {{
        font-size:20px !important;
        color: blue;
    }}
    .stButton>button {{
        background-color: #0000FF;
        color: white;
        width: 100%;
        text-align: left;
        padding: 5px;
        margin: 3px 0;
        font-size: 0.9em;
    }}
    </style>
    """, 
    unsafe_allow_html=True
)

# Load environment variables
load_dotenv()
# Initialize Anthropic client
anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Initialize Anthropic LLM (Claude 3.5 Sonnet)
llm = ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=0)

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j graph
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

# Create the GraphCypherQAChain
chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True
)

# Prompt template for generating Cypher queries
cypher_template = """
You are an AI expert specializing in converting natural language queries to Cypher queries for a Neo4j graph database in a medical context. Generate a precise and efficient Cypher query based on the given natural language query and the provided graph schema.

Graph Schema:
{schema}

Natural language query: {query}

Key Guidelines:
0. Always write your own Correlation function. NEVER try and use "corr()" in your Cypher queries.
1. Start with MATCH for pattern matching. Use a single WHERE clause for all conditions.
2. Use case-insensitive matching (=~ '(?i)...') for string comparisons.
3. Handle unit conversions and null values in CASE statements within WITH clauses.
4. Use OPTIONAL MATCH for potentially missing relationships.
5. Aggregate in the RETURN clause without using GROUP BY.
6. Always end with LIMIT 100 unless specified otherwise.
7. Ensure property names and relationships match the schema exactly.
8. Use relationships between nodes in both directions if more appropriate.
Specific Instructions:
1. For gender, use: p.Gender_txt =~ '(?i)male' or '(?i)female'
2. For age comparisons: toInteger(p.Age) > X
3. For BMI or other numeric comparisons: toFloat(p.BMI) > X
4. When searching names, locations, medications or any other text search always use a fuzzy match
5. Convert units when necessary (e.g., inches to cm, lbs to kg)
6. For date comparisons, use the following pattern:
   - apoc.date.parse(p.DateProperty, 'ms', 'yyyy-MM-dd') > apoc.date.parse('2022-01-01', 'ms', 'yyyy-MM-dd')
7. For date calculations or extractions, use:
   - apoc.date.field(apoc.date.parse(p.DateProperty, 'ms', 'yyyy-MM-dd'), 'year') as year
8. For numeric calculations on text fields, always use:
   - toFloat(p.NumericProperty) or toInteger(p.NumericProperty)
9. When aggregating or performing calculations on numeric fields stored as text, use:
   - avg(toFloat(p.NumericProperty)) as avgValue
10. For date ranges, use:
    - apoc.date.parse(p.DateProperty, 'ms', 'yyyy-MM-dd') >= apoc.date.parse('2022-01-01', 'ms', 'yyyy-MM-dd') AND 
      apoc.date.parse(p.DateProperty, 'ms', 'yyyy-MM-dd') < apoc.date.parse('2023-01-01', 'ms', 'yyyy-MM-dd')
11. Always alias all expressions and property accesses in WITH clauses, even if they're not transformed.
12. When using multiple WITH clauses, ensure all fields used in subsequent clauses are properly aliased in preceding clauses.
13. Use WHERE clauses early in the query to filter out null values before performing calculations or comparisons.

Critical Query Structure Guidelines:
1. Always alias ALL node properties in WITH clauses, even if they're not being transformed.
   Correct:   WITH n, n.property AS property_alias
   Incorrect: WITH n, n.property

2. Use early WHERE clauses to filter out null values before any calculations.
   Example: WHERE n.property1 IS NOT NULL AND n.property2 IS NOT NULL

3. In RETURN clauses, always use aliased property names from previous WITH clauses.
   Correct:   RETURN avg(toFloat(property_alias)) AS avg_property
   Incorrect: RETURN avg(toFloat(n.property)) AS avg_property

4. When using CASE statements in WITH clauses, always alias the result.
   Example: 
   WITH n, 
     CASE 
       WHEN condition THEN value1 
       ELSE value2 
     END AS case_result

5. For every property used in calculations or comparisons, ensure it's aliased in a WITH clause first.

6. When working with numeric properties stored as text:
   a. Always use toFloat() or toInteger() for conversion.
   b. Handle potential NULL or invalid text values using CASE statements.
   c. Use coalesce() to provide default values for NULL results in calculations.

Example of handling text-stored numeric properties:
MATCH (n:Node)
WHERE n.textProperty IS NOT NULL
WITH n,
  CASE
    WHEN n.textProperty =~ '^[0-9]+(\.[0-9]+)?$' 
    THEN toFloat(n.textProperty)
    ELSE NULL
  END AS numericValue
WITH 
  CASE 
    WHEN numericValue IS NOT NULL THEN 'Valid' 
    ELSE 'Invalid' 
  END AS valueStatus,
  numericValue
RETURN
  valueStatus,
  coalesce(avg(numericValue), 0) AS averageValue,
  count(*) AS totalCount

7. When working with numeric properties stored as text with units:
   a. Use SUBSTRING and REPLACE functions to extract the numeric part.
   b. Convert the extracted part to a number using toFloat() or toInteger().
   c. Handle null values and potential invalid formats.
   d. Use coalesce() to provide default values for NULL results in calculations.

Example of handling text-stored numeric properties with units:
MATCH (n:Node)
WITH n,
  CASE
    WHEN n.measurementProperty IS NULL THEN NULL
    WHEN n.measurementProperty =~ '^[0-9]+mm$' 
    THEN toInteger(REPLACE(n.measurementProperty, 'mm', ''))
    ELSE NULL
  END AS numericValue
WITH 
  CASE 
    WHEN numericValue IS NOT NULL THEN 'Valid' 
    ELSE 'Invalid' 
  END AS valueStatus,
  numericValue
RETURN
  valueStatus,
  coalesce(avg(numericValue), 0) AS averageValue,
  count(*) AS totalCount

Specific data handling for thickness measurements:
MATCH (fr:FemoralResection)
WHERE fr.DFRLateralCondyleStatus IS NOT NULL
WITH 
  fr,
  fr.DFRLateralCondyleStatus AS condyle_status,
  CASE
    WHEN fr.DFRLateralCondyleInitialThickness_txt =~ '^[0-9]+mm$'
    THEN toInteger(REPLACE(fr.DFRLateralCondyleInitialThickness_txt, 'mm', ''))
    ELSE NULL
  END AS initial_thickness,
  CASE
    WHEN fr.DFRLateralCondyleFinalThickness_txt =~ '^[0-9]+mm$'
    THEN toInteger(REPLACE(fr.DFRLateralCondyleFinalThickness_txt, 'mm', ''))
    ELSE NULL
  END AS final_thickness
WITH 
  condyle_status,
  initial_thickness,
  final_thickness
RETURN
  condyle_status,
  avg(initial_thickness) AS avg_initial_thickness,
  avg(final_thickness) AS avg_final_thickness,
  count(*) AS total_count
ORDER BY condyle_status
LIMIT 100;

8. Always include diagnostic counts in queries dealing with potentially problematic data:
   - Total count of records
   - Count of valid (non-NULL) converted values
   This helps identify potential data quality issues and conversion problems.

9. When dealing with averages of potentially problematic numeric data, avoid using coalesce() to replace NULL with 0, as this can skew results. Instead, let avg() handle NULLs naturally and provide count information for context.
10. Always carry forward all necessary variables in WITH clauses, especially node variables like patient (p) that are needed for later operations.
    Correct:   WITH p, otherVar1, otherVar2
    Incorrect: WITH otherVar1, otherVar2

11. When using OPTIONAL MATCH, ensure that the main entity (e.g., patient) is carried through all subsequent WITH clauses.

12. Before the final RETURN clause, use a WITH clause that explicitly lists all variables needed for the return, including those used in aggregate functions.

Example of correct variable handling in a multi-step query:

MATCH (p:Patient)
WITH p, toFloat(p.SomeProperty) AS someValue
OPTIONAL MATCH (p)-[:SOME_RELATIONSHIP]->(o:OtherNode)
WHERE o.SomeCondition IS NOT NULL
WITH p, someValue, CASE WHEN o.SomeCondition IS NOT NULL THEN 1 ELSE 0 END AS someCondition
RETURN
  count(DISTINCT p) AS totalPatients,
  avg(someValue) AS averageValue,
  sum(someCondition) AS conditionCount
LIMIT 100;
Critical Relationship Patterns:
1. Patient to Procedure: (p:Patient)-[:UNDERWENT]->(pr:Procedure)
2. Procedure to Location: (pr:Procedure)-[:PERFORMED_AT]->(l:Location)
3. Location to SurgicalTeam: (l:Location)-[:HAS_TEAM]->(st:SurgicalTeam)
4. Full path: (p:Patient)-[:UNDERWENT]->(pr:Procedure)-[:PERFORMED_AT]->(l:Location)-[:HAS_TEAM]->(st:SurgicalTeam)

13. When calculating ratios or percentages:
    a. Use CASE statements to create boolean values (0 or 1) for each category.
    b. Sum these values to get counts for each category.
    c. Calculate ratios using floating-point division.
    d. Use round() function to limit decimal places in the result.

Example of correct ratio calculation:
MATCH (p:Patient)
WITH 
  CASE WHEN p.Gender_txt =~ '(?i)male' THEN 1 ELSE 0 END AS is_male,
  CASE WHEN p.Gender_txt =~ '(?i)female' THEN 1 ELSE 0 END AS is_female
WITH 
  sum(is_male) AS male_count,
  sum(is_female) AS female_count,
  count(*) AS total_count
RETURN
  round(toFloat(male_count) / total_count, 2) AS male_ratio,
  round(toFloat(female_count) / total_count, 2) AS female_ratio,
  total_count

Common Pitfalls to Avoid:
1. Don't assume direct relationships between nodes that are not connected in the schema.
2. Always use COUNT(DISTINCT ...) when counting relationships to avoid overcounting.
3. Check for null values in critical fields before performing operations on them.
4. Use OPTIONAL MATCH for relationships that may not always exist to avoid losing data.

14. For queries involving patients and related entities, always ensure the patient node (typically aliased as 'p') is carried through all WITH clauses up to the final RETURN.
15. For patient-related queries, always use the structure provided in "Query Structure for Patient-Related Queries".
16. Avoid multiple WITH clauses in patient-related queries to prevent variable scoping issues.
17. Perform all calculations and case statements in a single WITH clause immediately before the RETURN statement.

Query Structure for Patient-Related Queries:
1. Start with MATCH (p:Patient) to establish the patient node.
2. Use a single WITH clause before the RETURN statement to prepare all necessary calculations and aggregations.
3. In this WITH clause, perform all necessary calculations and create case statements.
4. Use this structure for patient-related queries:
5 NEVER try and use "corr" function for correlation in Cypher queries. Always write the Correlation from first principles.

MATCH (p:Patient)
OPTIONAL MATCH (p)-[:RELATIONSHIP]->(:OtherNode)
WHERE <conditions>
WITH p,
  <calculation1> AS var1,
  <calculation2> AS var2,
  CASE WHEN <condition> THEN <value1> ELSE <value2> END AS var3
RETURN
  <aggregations and results>
LIMIT 100


This structure ensures that the patient node 'p' is available for all calculations and aggregations in the RETURN clause.
Output only the Cypher query, ending with a semicolon.
"""

cypher_prompt = PromptTemplate(
    input_variables=["schema", "query"],
    template=cypher_template
)

def execute_cypher_query(query):
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

def extract_cypher_query(generated_text):
    logging.info(f"Extracting Cypher query from: {generated_text[:100]}...")  # Log first 100 chars

    # Regular expression to match Cypher query, including multi-line
    cypher_pattern = r'(?s)(MATCH|MERGE|CREATE|CALL|UNWIND).*?;'
    
    match = re.search(cypher_pattern, generated_text, re.IGNORECASE | re.DOTALL)
    
    if match:
        full_query = match.group(0)
        logging.info(f"Found Cypher query: {full_query}")
        
        # Remove any comments
        full_query = re.sub(r'//.*?(\n|$)', '', full_query)
        full_query = re.sub(r'/\*[\s\S]*?\*/', '', full_query)
        
        # Fix EXISTS clauses (be more careful with this)
        full_query = re.sub(r'EXISTS\s*\(\s*(.+?)\s*\)\s*WHERE', r'EXISTS( \1 WHERE', full_query)
        
        # Clean up whitespace
        full_query = re.sub(r'\s+', ' ', full_query).strip()
        
        logging.info(f"Cleaned Cypher query: {full_query}")
        return full_query
    else:
        logging.warning("No valid Cypher query found in the generated text.")
        return None

class TokenCountingCallback(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.token_count = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        for prompt in prompts:
            self.token_count += len(prompt.split())

    def on_llm_end(self, response, **kwargs):
        for generation in response.generations:
            self.token_count += len(generation[0].text.split())

def display_langchain_api_info(api_info):
    sonnet_purple = "#6B46C1"
    sonnet_light_purple = "#9F7AEA"
    
    st.markdown(f"""
    <style>
        .api-info-langchain {{
            font-size: 0.9rem;
            color: {sonnet_purple};
        }}
        .api-info-langchain .metric-value {{
            font-weight: bold;
            color: {sonnet_light_purple};
        }}
        .api-info-langchain .footnote {{
            font-size: 0.7rem;
            color: #718096;
            font-style: italic;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    with st.expander("LangChain API Call Information", expanded=False):
        st.markdown('<div class="api-info-langchain">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Model Used:** <span class='metric-value'>{api_info['model']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Total Tokens:** <span class='metric-value'>{api_info['total_tokens']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Execution Time:** <span class='metric-value'>{api_info['execution_time']:.2f} seconds</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**Cost:** <span class='metric-value'>${api_info['cost']:.6f}</span>", unsafe_allow_html=True)
        
        st.markdown('<p class="footnote">Note: Costs are approximate and based on Claude 3 Sonnet pricing. Token count and execution times may vary.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def verify_neo4j_connection():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        return False, "Neo4j environment variables are not set correctly."

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            with driver.session() as session:
                result = session.run("RETURN 1 AS test")
                result.single()
        return True, "Successfully connected to Neo4j database."
    except Exception as e:
        return False, f"Failed to connect to Neo4j database: {str(e)}"

# Add this to your main function or Streamlit app
connection_success, connection_message = verify_neo4j_connection()
if not connection_success:
    st.error(connection_message)
else:
    st.success(connection_message)


def graph_qa_page():
    st.title("Graph Question and Answer")

    # Initialize the database
    init_db()

    # Initialize session state variables
    if 'selected_action' not in st.session_state:
        st.session_state.selected_action = None
    if 'query_executed' not in st.session_state:
        st.session_state.query_executed = False
    if 'clear_input' not in st.session_state:
        st.session_state.clear_input = False

    # Main buttons at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear", key="clear_top"):
            st.session_state.selected_action = None
            st.session_state.query_executed = False
            st.session_state.user_query = ""
            st.rerun()
    with col2:
        if st.button("Run Pre-Configured Query"):
            st.session_state.selected_action = "preconfigured"
    with col3:
        if st.button("Ask a New Question"):
            st.session_state.selected_action = "new_question"

    if st.session_state.selected_action == "preconfigured":
        run_preconfigured_query()
    elif st.session_state.selected_action == "new_question":
        ask_new_question_flow()

def run_preconfigured_query():
    queries = get_preconfigured_queries()
    if queries:
        selected_query = st.selectbox("Select a pre-configured query:", 
                                      [f"{q[0]}: {q[1]}" for q in queries])
        if selected_query:
            query_id = int(selected_query.split(":")[0])
            conn = sqlite3.connect('Medacta.db')
            c = conn.cursor()
            c.execute("SELECT cypher_query FROM Graph_queries WHERE id = ?", (query_id,))
            cypher_query = c.fetchone()[0]
            conn.close()
            
            # Execute the selected Cypher query
            result = execute_cypher_query(cypher_query)
            if result is not None:
                st.subheader("Query Result:")
                st.write(result)
                
                # Expandable box for the Cypher query
                with st.expander("View Cypher Query", expanded=False):
                    st.code(cypher_query, language="cypher")
                
                st.session_state.query_executed = True
            else:
                st.error("Error executing the Cypher query.")
    else:
        st.warning("No pre-configured queries available.")

def ask_new_question_flow():
    # Get graph schema
    schema_json = get_graph_schema()
    schema = json.loads(schema_json)

    if "error" in schema:
        st.error(f"Error fetching schema: {schema['error']}")
        return

    # Display the schema in an expander
    with st.expander("View Graph Schema", expanded=False):
        st.json(schema)

    # User input
    user_query = st.text_input("Enter your question about the graph data:", key="user_query")

    if user_query:
        try:
            # Initialize the token counting callback
            token_callback = TokenCountingCallback()

            # Reinitialize the LLM and chains with the callback
            llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, callbacks=[token_callback])
            
            cypher_chain = LLMChain(llm=llm, prompt=cypher_prompt)
            chain = GraphCypherQAChain.from_llm(
                llm,
                graph=graph,
                verbose=True,
                cypher_prompt=cypher_prompt
            )

            # Generate Cypher query
            cypher_generation_start = time.time()
            generated_text = cypher_chain.run(schema=schema_json, query=user_query)
            cypher_generation_end = time.time()

            print("Generated text:", generated_text)
            logging.info(f"Generated text: {generated_text}")

            # Extract valid Cypher query
            cypher_query = generated_text
            print("Extracted Cypher query:", cypher_query)
            logging.info(f"Extracted Cypher query: {cypher_query}")

            if cypher_query is None:
                st.warning("Unable to generate a valid Cypher query based on the provided schema and query.")
                st.error("Generated content:")
                st.code(generated_text)
            else:
                # Execute Cypher query
                execution_start = time.time()
                try:
                    result = execute_cypher_query(cypher_query)
                    execution_end = time.time()

                    if result is None:
                        st.error("Error executing the Cypher query. Please check the Neo4j connection and query syntax.")
                    elif len(result) == 0:
                        st.warning("The query executed successfully, but returned no results.")
                        # Expandable box for the generated text (including explanations)
                        with st.expander("View Generated Text", expanded=False):
                            st.code(generated_text, language="markdown")

                        # Expandable box for the extracted Cypher query
                        with st.expander("View Extracted Cypher Query", expanded=False):
                            st.code(cypher_query, language="cypher")
                    else:
                        # Display the result
                        st.subheader("Query Result:")
                        st.write(result)

                        # Expandable box for the full prompt
                        with st.expander("View Full Prompt", expanded=False):
                            st.code(cypher_prompt.format(schema=schema_json, query=user_query), language="markdown")

                        # Expandable box for the generated text (including explanations)
                        with st.expander("View Generated Text", expanded=False):
                            st.code(generated_text, language="markdown")

                        # Expandable box for the extracted Cypher query
                        with st.expander("View Extracted Cypher Query", expanded=False):
                            st.code(cypher_query, language="cypher")

                        # Calculate total execution time
                        total_time = (cypher_generation_end - cypher_generation_start) + \
                                     (execution_end - execution_start)

                        # Prepare API info using the callback data
                        total_tokens = token_callback.token_count
                        api_info = {
                            "model": "claude-3-sonnet-20240229",
                            "total_tokens": total_tokens,
                            "execution_time": total_time,
                            "cost": calculate_cost(total_tokens, total_tokens)
                        }

                        display_langchain_api_info(api_info)

                        # Buttons for Explain Query and Add to KnowledgeHub
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Explain Query", key="explain_query"):
                                explanation_prompt = f"""
                                Please explain the following Cypher query in plain English. 
                                Break down each part of the query and describe what it's doing in simple terms.
                                Summarise with a description in simple terms of what the query shows.
                                
                                Cypher Query:
                                {cypher_query}
                                
                                Explanation:
                                """
                                
                                explanation = llm.predict(explanation_prompt)
                                
                                with st.expander("Query Explanation", expanded=True):
                                    st.write(explanation)

                        with col2:
                            if st.button("Add Question to KnowledgeHub", key="add_to_knowledgehub"):
                                add_query_to_db(user_query, cypher_query)
                                st.success("Query added to KnowledgeHub successfully!")

                    st.session_state.query_executed = True

                except Exception as e:
                    st.error(f"Error executing Cypher query: {str(e)}")
                    st.code(cypher_query, language="cypher")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Error in graph_qa_page: {str(e)}", exc_info=True)

def init_db():
    conn = sqlite3.connect('Medacta.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Graph_queries
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  question TEXT,
                  cypher_query TEXT)''')
    conn.commit()
    conn.close()

def add_query_to_db(question, cypher_query):
    conn = sqlite3.connect('Medacta.db')
    c = conn.cursor()
    c.execute("INSERT INTO Graph_queries (question, cypher_query) VALUES (?, ?)", (question, cypher_query))
    conn.commit()
    conn.close()

def get_preconfigured_queries():
    conn = sqlite3.connect('Medacta.db')
    c = conn.cursor()
    c.execute("SELECT id, question FROM Graph_queries")
    queries = c.fetchall()
    conn.close()
    return queries

def delete_graph_queries():
    st.subheader("Delete Graph Queries")

    # Fetch all queries
    conn = sqlite3.connect('Medacta.db')
    df = pd.read_sql_query("SELECT id, question, cypher_query FROM Graph_queries", conn)
    conn.close()

    if df.empty:
        st.warning("No queries found in the database.")
        return

    # Add a checkbox column
    df['Delete'] = False

    # Display the dataframe as an editable table
    edited_df = st.data_editor(
        df,
        hide_index=True,
        column_config={
            "Delete": st.column_config.CheckboxColumn(
                "Select",
                help="Select to delete",
                default=False,
            ),
        
            "question": st.column_config.TextColumn(
                "Question",
                help="The natural language question",
                max_chars=50,
            ),
            "cypher_query": st.column_config.TextColumn(
                "Cypher Query",
                help="The corresponding Cypher query",
                max_chars=50,
            ),
        },
        disabled=["id", "question", "cypher_query"],
        num_rows="dynamic",
    )

    # Delete button
    if st.button("Delete Selected Queries"):
        selected_ids = edited_df[edited_df['Delete']]['id'].tolist()
        
        if not selected_ids:
            st.warning("No queries selected for deletion.")
        else:
            try:
                conn = sqlite3.connect('Medacta.db')
                c = conn.cursor()
                c.executemany("DELETE FROM Graph_queries WHERE id = ?", [(id,) for id in selected_ids])
                conn.commit()
                conn.close()
                st.success(f"{len(selected_ids)} queries deleted successfully.")
                
                # Set a flag in session state to indicate deletion occurred
                st.session_state.queries_deleted = True
                
                # Use st.rerun() to refresh the page
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred while deleting queries: {str(e)}")

    # Provide a back button to return to the main interface
    if st.button("Refresh"):
        st.session_state.page = "main"  # Assuming you have a way to navigate back to the main page
        st.rerun()


def set_page_background(color):
    page_bg_img = f'''
    <style>
    .stApp {{
        background-color: {color};
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Database functions
def get_questions_from_db():
    conn = sqlite3.connect('Medacta.db')
    cursor = conn.cursor()
    cursor.execute("SELECT question FROM questions")
    questions = [row[0] for row in cursor.fetchall()]
    conn.close()
    return questions

def get_schema_info_from_db():
    conn = sqlite3.connect('Medacta.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM schema_info")
    schema_info = cursor.fetchall()
    conn.close()
    return schema_info

def get_data(query, params=None):
    conn = sqlite3.connect('medacta.db')
    if params:
        df = pd.read_sql_query(query, conn, params=params)
    else:
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def execute_sql_query(query):
    conn = sqlite3.connect('Medacta.db')
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        return str(e)

def add_question_to_db(question):
    conn = sqlite3.connect('Medacta.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO questions (question) VALUES (?)", (question,))
    conn.commit()
    conn.close()

def get_eav_data(patient_pid):
    conn = sqlite3.connect('Medacta.db')
    cursor = conn.cursor()
    cursor.execute("SELECT Attribute, Value FROM Surgical_Data_EAV WHERE EntityID = ?", (patient_pid,))
    eav_data = cursor.fetchall()
    conn.close()
    return eav_data

def get_all_attributes():
    conn = sqlite3.connect('Medacta.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Attribute FROM Surgical_Data_EAV ORDER BY Attribute")
    attributes = [row[0] for row in cursor.fetchall()]
    conn.close()
    return attributes

def get_attribute_values(attribute):
    conn = sqlite3.connect('Medacta.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Value FROM Surgical_Data_EAV WHERE Attribute = ? ORDER BY Value", (attribute,))
    values = [row[0] for row in cursor.fetchall()]
    conn.close()
    return values

def get_pathways_for_attribute_value(attribute, value):
    conn = sqlite3.connect('Medacta.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT EntityID 
        FROM Surgical_Data_EAV 
        WHERE Attribute = ? AND Value = ?
        ORDER BY EntityID
    """, (attribute, value))
    pathways = [row[0] for row in cursor.fetchall()]
    conn.close()
    return pathways

# Helper functions
def fuzzy_search(query, choices, limit=5):
    return process.extract(query, choices, limit=limit)

def generate_prompt_template(question, matched_attributes, user_constraints):
    # Read the content of Short_Desc.txt
    short_desc_path = "Short_Desc.txt"
    short_desc_content = ""
    if os.path.exists(short_desc_path):
        with open(short_desc_path, "r") as file:
            short_desc_content = file.read()
    else:
        print("Warning: Short_Desc.txt file not found. Proceeding without it.")

    prompt = f"""You are an AI assistant specialized in translating natural language queries into SQL for a medical database called medacta.db. Your task is to generate SQL queries based on user input while adhering to specific constraints. You will be using the table Surgical_data_EAV. Very importantL The Surgical_data_EAV table is in an Entity-Attribute-Value structure so please handle with that in mind at all times.

Given the following question: "{question}"
And the following matched attributes from the Surgical_data_EAV table:
{', '.join(matched_attributes)}
Please generate an SQL query that satisfies the following requirements:
1. Use only the fields available in the Surgical_data_EAV table.
2. Use .txt fields of the corresponding attribute where necessary but especially when measurements are being referred to such as thicknesses, diameters, lengths, etc.
3. The query should aim to answer the original question as accurately as possible.
4. Always Return the Query result by grouping entityID unless the question specifically indicates not to.
5. ALWAYS filter the Matching Attributes down from the list provided to just the ones that are relevant to the Question being asked. Use the Short Desc document provided to do this. This is important or you will create a more complicated query than is required.
6. ALWAYS use "LIKE" when querying text fields

Here is the content of the Short_Desc.txt document for reference:
{short_desc_content}

Additional constraints provided by the user:
{user_constraints}

Generate an SQL query that meets these criteria:
"""
    return prompt


def show_prompt_screen():
    st.header("SQL Query Generator")
    
    # Initialize session state variables if they don't exist
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if 'additional_rules' not in st.session_state:
        st.session_state.additional_rules = ""

    # Display General Rules
    general_rules = """General Rules:
1. Only use fields that exist in the provided database schema.
2. If the user input doesn't translate to a sensible query, return "No query formed".
3. Ensure the query is syntactically correct for SQLite.
4. NEVER use any fields or tables not explicitly defined in the schema.
5. Always use the field CompleteDate when compiling your query.
6. Once you have interpreted the DB field to use ALWAYS use the .txt field version as these store the true values.
7. Always use the definitions provided to find the right query fields.
8. Ensure the query always returns a result set (i.e., avoid queries that return only a single value or perform operations without returning data).
9. ALWAYS use "LIKE" when querying text fields
10. Use user friendly Display Names when you present the data
11. Patient is synonymous with PathwayID in the DB
12. Where there are a range of values in the fields being queried. I want the results to be by EACH of these values."""

    st.text_area("General Rules:", value=general_rules, height=200, disabled=True)

    # User input for additional rules
    additional_rules = st.text_area("Enter additional rules (optional):", 
                                    value=st.session_state.additional_rules,
                                    height=100, 
                                    key="additional_rules_area")
    st.markdown('<style>textarea#additional_rules_area{color: blue;}</style>', unsafe_allow_html=True)

    # User input for query
    user_input = st.text_area("Enter your query in natural language:", 
                              height=150, 
                              key="user_input_area", 
                              value=st.session_state.user_input)
    
    explain = st.checkbox("Explain the generated query", key="explain_checkbox")

    if st.button("Generate SQL", key="generate_sql_button"):
        if user_input:
            with st.spinner("Generating SQL query..."):
                result = generate_sql_query(user_input, additional_rules, explain)
                if explain:
                    if len(result) == 3:
                        sql_query, explanation, api_info = result
                        st.code(sql_query, language="sql")
                        st.subheader("Explanation:")
                        st.write(explanation)
                    else:
                        st.error("Unexpected result format when explanation was requested.")
                        return
                else:
                    if len(result) == 2:
                        sql_query, api_info = result
                        st.code(sql_query, language="sql")
                    else:
                        st.error("Unexpected result format.")
                        return
                
                display_api_info(api_info)
                
                if sql_query != "No query formed":
                    try:
                        df = execute_query(sql_query)
                        st.write("Query Results:")
                        st.dataframe(df)
                    except Exception as e:
                        st.error(f"Error executing query: {str(e)}")
        else:
            st.warning("Please enter a query before submitting.")
    
    if st.button("Clear", key="clear_button"):
        st.session_state.user_input = ""
        st.session_state.additional_rules = ""
        st.rerun()

    # Update session state
    st.session_state.user_input = user_input
    st.session_state.additional_rules = additional_rules

def generate_sql_query(user_input, additional_rules, explain=False):
    # Read the database schema from the short_desc.txt file
    schema_file_path = "DataBase Schema for US Data.txt"
    try:
        with open(schema_file_path, "r") as schema_file:
            database_schema = schema_file.read()
    except FileNotFoundError:
        database_schema = "Error: Database schema file 'DataBase Schema for US Data.txt' not found."
    except Exception as e:
        database_schema = f"Error reading database schema file: {str(e)}"

    prompt = f"""You are an AI assistant specialized in translating natural language queries into SQL for a medical database called medacta.db. Your task is to generate SQL queries based on user input.

Database Schema:
{database_schema}

User Input:
{user_input}

1. Only use fields that exist in the provided database schema.
2. If the user input doesn't translate to a sensible query, return "No query formed".
3. Ensure the query is syntactically correct for SQLite.
4. NEVER use any fields or tables not explicitly defined in the schema.
5. Always use the field CompleteDate when compiling your query.
6. Once you have interpreted the DB field to use ALWAYS use the .txt field version as these store the true values.
7. Always use the definitions provided to find the right query fields.
8. Ensure the query always returns a result set (i.e., avoid queries that return only a single value or perform operations without returning data).
9. ALWAYS use "LIKE" when querying text fields
10. Use user friendly Display Names when you present the data
11. Patient is synonymous with PathwayID in the DB
12. Where there are a range of values in the fields being queried. I want the results to be by EACH of these values.

Additional Rules:
{additional_rules}

Your response should be ONLY the SQL query, without any additional explanation or context, unless explicitly requested.

If you cannot form a valid query based on the input, respond with "No query formed".

SQL Query:
"""

    anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    start_time = time.time()
    response = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    end_time = time.time()
    execution_time = end_time - start_time

    sql_query = response.content[0].text.strip()
    api_info = {
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost": calculate_cost(response.usage.input_tokens, response.usage.output_tokens),
        "execution_time": execution_time
    }

    if explain and sql_query != "No query formed":
        explanation_prompt = f"""
Please explain the following SQL query and how it relates to the user's input:

User Input:
{user_input}

Generated SQL Query:
{sql_query}

Explanation:
"""
        explanation_start_time = time.time()
        explanation_response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": explanation_prompt}
            ]
        )
        explanation_end_time = time.time()
        explanation_execution_time = explanation_end_time - explanation_start_time
        
        api_info["explanation_input_tokens"] = explanation_response.usage.input_tokens
        api_info["explanation_output_tokens"] = explanation_response.usage.output_tokens
        api_info["explanation_cost"] = calculate_cost(explanation_response.usage.input_tokens, explanation_response.usage.output_tokens)
        api_info["explanation_execution_time"] = explanation_execution_time
        api_info["total_cost"] = api_info["cost"] + api_info["explanation_cost"]
        api_info["total_execution_time"] = execution_time + explanation_execution_time
        
        return sql_query, explanation_response.content[0].text.strip(), api_info
    else:
        return sql_query, api_info

def calculate_cost(input_tokens, output_tokens):
    # Claude 3 Sonnet pricing: $3 per 1M input tokens, $15 per 1M output tokens
    input_cost = (input_tokens / 1_000_000) * 3
    output_cost = (output_tokens / 1_000_000) * 15
    return input_cost + output_cost

def display_api_info(api_info):
    # Define Sonnet's brand colors
    sonnet_purple = "#6B46C1"
    sonnet_light_purple = "#9F7AEA"
    
    # Custom CSS for styling
    st.markdown(f"""
    <style>
        .api-info {{
            font-size: 0.9rem;
            color: {sonnet_purple};
        }}
        .api-info .metric-value {{
            font-weight: bold;
            color: {sonnet_light_purple};
        }}
        .api-info .footnote {{
            font-size: 0.7rem;
            color: #718096;
            font-style: italic;
        }}
    </style>
    """, unsafe_allow_html=True)

    with st.expander("API Call Information", expanded=False):
        st.markdown('<div class="api-info">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Model Used:** <span class='metric-value'>{api_info['model']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Input Tokens:** <span class='metric-value'>{api_info['input_tokens']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Output Tokens:** <span class='metric-value'>{api_info['output_tokens']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Execution Time:** <span class='metric-value'>{api_info['execution_time']:.2f} seconds</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**Cost:** <span class='metric-value'>${api_info['cost']:.6f}</span>", unsafe_allow_html=True)
            
            if "explanation_input_tokens" in api_info:
                st.markdown(f"**Explanation Input Tokens:** <span class='metric-value'>{api_info['explanation_input_tokens']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Explanation Output Tokens:** <span class='metric-value'>{api_info['explanation_output_tokens']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Explanation Cost:** <span class='metric-value'>${api_info['explanation_cost']:.6f}</span>", unsafe_allow_html=True)
                st.markdown(f"**Explanation Time:** <span class='metric-value'>{api_info['explanation_execution_time']:.2f} seconds</span>", unsafe_allow_html=True)
                st.markdown(f"**Total Cost:** <span class='metric-value'>${api_info['total_cost']:.6f}</span>", unsafe_allow_html=True)
                st.markdown(f"**Total Time:** <span class='metric-value'>{api_info['total_execution_time']:.2f} seconds</span>", unsafe_allow_html=True)

        st.markdown('<p class="footnote">Note: Costs are approximate and based on Claude 3 Sonnet pricing. Execution times may vary.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def generate_summary(patient_id, eav_data):
    attribute_value_pairs = "\n".join([f"{row['Attribute']}: {row['Value']}" for _, row in eav_data.iterrows()])
    
    prompt = f"""You are an experienced writer and summariser. I would like a Procedure Report which takes the data collected for a specific PathwayPID and provides a summary of the procedure. Start with a paragraph of the patient and a general overview of the procedure. I would like you to use all your knowledge to use the data and provide a  comprehensive summary with a good level of patient and technical narrative so that I can read these notes in the future and understood fully the procedure that was undertaken. I would also like you to provide an indication of your expectation of the outcome for this patient taking into consideration all the factors you have seen. Use your training to add context. Please provide your outcome as a grade of Expected Patient Outcome.

PathwayPID: {patient_id}

Attribute-Value Pairs:
{attribute_value_pairs}

Please provide a comprehensive summary based on this information."""

    response = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

def execute_query(query):
    conn = sqlite3.connect('Medacta.db')
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        conn.close()
        return f"SQLite error: {str(e)}"
    except Exception as e:
        conn.close()
        return f"An error occurred: {str(e)}"

def save_text_to_pdf(text, filename):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    for line in text.split('\n'):
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line)
        y -= 15
    c.save()
    buffer.seek(0)
    return buffer


def show_surgeries_by_surgeon():
    st.header("Number of Surgeries by Surgeon")

    min_date = pd.to_datetime(get_data("SELECT MIN(CompleteDate) FROM Surgical_Data").iloc[0, 0])
    max_date = pd.to_datetime(get_data("SELECT MAX(CompleteDate) FROM Surgical_Data").iloc[0, 0])

    st.write(f"Date range in database: {min_date.date()} to {max_date.date()}")

    date_range = st.slider(
        "Select date range",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date()),
        format="YYYY-MM-DD"
    )
    start_date, end_date = date_range

    surgeon_query = """
    SELECT 
        Clinician_txt AS Surgeon,
        COUNT(*) AS SurgeryCount
    FROM 
        Surgical_Data
    WHERE 
        date(CompleteDate) BETWEEN ? AND ?
    GROUP BY 
        Clinician_txt
    ORDER BY 
        SurgeryCount DESC
    """
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    surgeon_data = get_data(surgeon_query, params=(start_date_str, end_date_str))

    st.write("Number of surgeries performed by each surgeon:")
    st.dataframe(surgeon_data)

    fig = px.bar(surgeon_data, x='Surgeon', y='SurgeryCount', 
                 title='Number of Surgeries by Surgeon',
                 labels={'SurgeryCount': 'Number of Surgeries'})
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig)


def show_distal_femoral_resection_dashboard():
    st.title("Distal Femoral Resection")

    # Date range selector
    min_date = pd.to_datetime(get_data("SELECT MIN(CompleteDate) FROM Surgical_Data").iloc[0, 0])
    max_date = pd.to_datetime(get_data("SELECT MAX(CompleteDate) FROM Surgical_Data").iloc[0, 0])
    date_range = st.slider(
        "Select date range",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date()),
        format="YYYY-MM-DD"
    )

    # Get min and max values for Age and BMI
    age_min, age_max = get_data("SELECT MIN(Age), MAX(Age) FROM Surgical_Data").iloc[0]
    bmi_min, bmi_max = get_data("SELECT MIN(BMI), MAX(BMI) FROM Surgical_Data").iloc[0]

    # Controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age_range = st.slider("Age Range", int(age_min), int(age_max), (int(age_min), int(age_max)))
    with col2:
        bmi_range = st.slider("BMI Range", float(bmi_min), float(bmi_max), (float(bmi_min), float(bmi_max)))
    with col3:
        condyle_status = st.radio("Condyle Status", ["All", "Worn", "Unworn"])
    with col4:
        gender = st.radio("Gender", ["All", "Male", "Female"])

    # Prepare WHERE clause
    where_clause = f"WHERE date(CompleteDate) BETWEEN ? AND ? AND Age BETWEEN ? AND ? AND BMI BETWEEN ? AND ?"
    params = [date_range[0], date_range[1], age_range[0], age_range[1], bmi_range[0], bmi_range[1]]
    
    if condyle_status != "All":
        where_clause += f" AND DFRMedialCondyleStatus = ? AND DFRLateralCondyleStatus = ?"
        params.extend([1 if condyle_status == "Worn" else 2] * 2)
    
    if gender != "All":
        where_clause += f" AND Gender = ?"
        params.append(1 if gender == "Male" else 2)

    # Create two columns for the dashboard
    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Medial Condyle")
        create_charts("Medial", where_clause, params)

    with right_col:
        st.subheader("Lateral Condyle")
        create_charts("Lateral", where_clause, params)


def create_charts(side, where_clause, base_params):
    charts = [
        ("Initial Thickness", f"DFR{side}CondyleInitialThickness", f"DFR{side}CondyleInitialThickness_txt"),
        ("Final Thickness", f"DFR{side}CondyleFinalThickness", f"DFR{side}CondyleFinalThickness_txt"),
        ("Recut Amount", f"DFR{side}CondyleRecut", None),
        ("Washer Size", f"DFR{side}CondyleWasher", None)
    ]

    for title, field, txt_field in charts:
        if txt_field:
            query = f"""
            SELECT 
                {txt_field} AS Measurement,
                COUNT(*) AS PatientCount
            FROM 
                Surgical_Data
            {where_clause}
            AND {txt_field} != ''
            GROUP BY 
                {txt_field}
            ORDER BY 
                {field}
            """
            params = base_params
        else:
            x_values = ["1", "2", "3", "4", "5"]
            query = f"""
            SELECT 
                CAST({field} AS TEXT) AS Measurement,
                COUNT(*) AS PatientCount
            FROM 
                Surgical_Data
            {where_clause}
            AND {field} IN ({','.join(['?']*len(x_values))})
            GROUP BY 
                {field}
            ORDER BY 
                {field}
            """
            params = base_params + x_values

        data = get_data(query, params=params)
        
        if not data.empty:
            fig = px.bar(data, x='Measurement', y='PatientCount', title=f"{side} Condyle {title}",
                         color_discrete_sequence=['#40E0D0'])  # Turquoise color
            fig.update_layout(
                xaxis_title=title,
                yaxis_title="Number of Patients",
                xaxis_title_font=dict(size=10),
                yaxis_title_font=dict(size=10),
                bargap=0.2,
                bargroupgap=0.1
            )
            
            if txt_field:
                # For Initial and Final Thickness, use the labels from the data
                x_values = data['Measurement'].tolist()
                fig.update_xaxes(categoryorder='array', categoryarray=x_values)
            else:
                # For Recut and Washer Size, use the hardcoded values and add 'mm'
                fig.update_xaxes(categoryorder='array', categoryarray=x_values)
                fig.update_xaxes(ticktext=[f"{val} mm" for val in x_values], tickvals=x_values)
            
            # Format y-axis to show only integers
            fig.update_yaxes(tickformat="d")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"No data available for {side} Condyle {title}")

def main_app():
    if 'selected_question' not in st.session_state:
        st.session_state.selected_question = None
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'matched_attributes' not in st.session_state:
        st.session_state.matched_attributes = []
    if 'user_constraints' not in st.session_state:
        st.session_state.user_constraints = ""
    
    questions = get_questions_from_db()
    schema_info = get_schema_info_from_db()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="big-font">Question Search</p>', unsafe_allow_html=True)
        search_query = st.text_input("Start typing your question:", value=st.session_state.search_query, key="search_input")
        st.session_state.search_query = search_query
    
    with col2:
        st.markdown('<p class="big-font">Matching Questions</p>', unsafe_allow_html=True)
        if search_query:
            matches = fuzzy_search(search_query, questions)
            for match, score in matches:
                if st.button(match, key=match, help="Click to select this question"):
                    st.session_state.selected_question = match
                    st.rerun()
    
    if st.session_state.selected_question:
        st.markdown('<p class="big-font">Selected Question</p>', unsafe_allow_html=True)
        st.write(st.session_state.selected_question)
        
        # Make Matched Attributes section expandable and collapsible
        with st.expander("Matched Attributes", expanded=False):
            descriptions = {row[0]: row[1] for row in schema_info}
            matches = fuzzy_search(st.session_state.selected_question, list(descriptions.keys()), limit=10)
            
            st.session_state.matched_attributes = []
            for attribute, score in matches:
                tag = "🔴" if score > 50 else ""
                st.markdown(f"""
                    <div style="border: 1px solid #ddd; padding: 5px; margin: 3px 0; border-radius: 3px; font-size: 0.9em; position: relative;">
                        <h5 style="margin: 0; font-size: 1em;">{attribute} {tag}</h5>
                        <p style="margin: 0; color: blue; font-size: 0.9em;">{descriptions[attribute]}</p>
                        <div style="position: absolute; top: 0; right: 0; width: 10px; height: 100%; background-color: {'red' if score > 50 else 'transparent'};"></div>
                    </div>
                """, unsafe_allow_html=True)
                st.session_state.matched_attributes.append(attribute)
        
        st.markdown('<p class="big-font">SQL Generation</p>', unsafe_allow_html=True)
        user_constraints = st.text_area("Additional constraints for SQL generation:", value=st.session_state.user_constraints)
        st.session_state.user_constraints = user_constraints
        
        if st.button("Generate Prompt Text"):
            prompt = generate_prompt_template(
                st.session_state.selected_question,
                st.session_state.matched_attributes,
                st.session_state.user_constraints
            )
            st.text_area("Generated Prompt Template:", prompt, height=300)

        st.markdown('<p class="big-font">Execute Custom SQL Query</p>', unsafe_allow_html=True)
        custom_query = st.text_area("Enter your SQL query:", height=150)
        if st.button("Execute Query"):
            result = execute_query(custom_query)
            if isinstance(result, pd.DataFrame):
                st.markdown('<p class="big-font">Query Results</p>', unsafe_allow_html=True)
                st.dataframe(result)
            else:
                st.error(f"Error executing query: {result}")

    # Add a button to show all questions
    if st.button("Show All Questions"):
        show_all_questions()

def show_all_questions():
    questions = get_questions_from_db()
    if questions:
        with st.expander("All Questions", expanded=False):
            for idx, question in enumerate(questions, 1):
                st.write(f"{idx}. {question}")
    else:
        st.info("No questions found in the database.")

def show_all_data():
    st.header("All Surgical Data")
    
    # Fetch all data first
    all_data = get_data("SELECT * FROM Surgical_Data")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Ensure PathwayPID is the first column
    columns = ['PathwayPID'] + [col for col in df.columns if col != 'PathwayPID']
    df = df[columns]
    
    # Column selection
    st.subheader("Select Columns to Display")
    default_columns = list(df.columns[:5])  # Default to first 5 columns including PathwayPID
    selected_columns = st.multiselect(
        "Choose columns",
        options=list(df.columns[1:]),  # Exclude PathwayPID from options
        default=default_columns[1:]  # Default selection excludes PathwayPID
    )
    
    # Always include PathwayPID and add it to the beginning of the list
    selected_columns = ['PathwayPID'] + selected_columns
    
    # Filter data based on selected columns
    filtered_data = df[selected_columns]
    
    # Sorting options
    st.subheader("Sort Data")
    sort_column = st.selectbox("Select column to sort by", options=selected_columns)
    sort_order = st.radio("Sort order", options=["Ascending", "Descending"])
    
    # Apply sorting
    filtered_data = filtered_data.sort_values(by=sort_column, ascending=(sort_order == "Ascending"))
    
    # Configure AG Grid
    gb = GridOptionsBuilder.from_dataframe(filtered_data)
    gb.configure_default_column(sortable=True, filter=True, resizable=True)
    gb.configure_selection(selection_mode='multiple', use_checkbox=True)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_column("PathwayPID", pinned="left")  # Pin PathwayPID to the left
    gridOptions = gb.build()
    
    # Display the AG Grid
    st.subheader("Filtered and Sorted Data")
    grid_response = AgGrid(
        filtered_data, 
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )
    
    # Get results from AG Grid
    selected_rows = grid_response['selected_rows']
    displayed_data = grid_response['data']
    
    # Download filtered and sorted data as CSV
    csv = displayed_data.to_csv(index=False)
    st.download_button(
        label="Download displayed data as CSV",
        data=csv,
        file_name="filtered_sorted_data.csv",
        mime="text/csv",
    )
    
    # Display information about selected rows
    if selected_rows is not None and len(selected_rows) > 0:
        st.subheader("Selected Rows")
        st.write(pd.DataFrame(selected_rows))
    else:
        st.info("No rows selected. Select rows by clicking on them in the grid above.")

    # Debug information
    st.subheader("Debug Information")
    st.write("Grid Response Keys:", grid_response.keys())
    st.write("Selected Rows Type:", type(selected_rows))
    if selected_rows is not None:
        st.write("Selected Rows Length:", len(selected_rows))
    else:
        st.write("Selected Rows is None")

def eav_searcher():
    st.header("EAV Searcher")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Search by Attribute and Value")
        if 'selected_attribute' not in st.session_state:
            st.session_state.selected_attribute = ""
        if 'selected_value' not in st.session_state:
            st.session_state.selected_value = ""
        if st.button("Reset Attribute Search", key="reset_attribute_button"):
            st.session_state.selected_attribute = ""
            st.session_state.selected_value = ""
            st.rerun()
        
        attributes = get_all_attributes()
        selected_attribute = st.selectbox("Select Attribute:", options=[""] + attributes, index=0, key="attribute_select")
        st.session_state.selected_attribute = selected_attribute

        if selected_attribute:
            attribute_values = get_attribute_values(selected_attribute)
            selected_value = st.selectbox("Select Value:", options=[""] + attribute_values, index=0, key="value_select")
            st.session_state.selected_value = selected_value

            if selected_value:
                pathways = get_pathways_for_attribute_value(selected_attribute, selected_value)
                if pathways:
                    with st.expander(f"Pathways with {selected_attribute} = {selected_value} (Total: {len(pathways)})", expanded=False):
                        for pathway in pathways:
                            if st.button(f"PathwayPID: {pathway}", key=f"pathway_{pathway}"):
                                st.session_state.patient_id = pathway
                                st.rerun()
                else:
                    st.warning(f"No pathways found with {selected_attribute} = {selected_value}")

    with col2:
        st.subheader("Search by Pathway ID")
        if 'patient_id' not in st.session_state:
            st.session_state.patient_id = ""
        if st.button("Reset Pathway Search", key="reset_pathway_button"):
            st.session_state.patient_id = ""
            st.rerun()
        patient_id = st.text_input("Enter Patient ID:", value=st.session_state.patient_id)
        st.session_state.patient_id = patient_id
        
        if patient_id:
            eav_data = get_eav_data(patient_id)
            if eav_data:
                with st.expander(f"Results for Pathway ID: {patient_id}", expanded=False):
                    df = pd.DataFrame(eav_data, columns=['Attribute', 'Value'])
                    
                    for _, row in df.iterrows():
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 5px; border-radius: 5px;">
                            <strong>{row['Attribute']}:</strong> {row['Value']}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning(f"No data found for the given Pathway ID: {patient_id}")
                st.info("Please check the ID and try again. If the problem persists, the table structure might be different from what's expected.")

    # Add the "Provide Pathway Summary" button
    if st.button("Provide Pathway Summary", key="summary_button"):
        if st.session_state.patient_id:
            eav_data = get_eav_data(st.session_state.patient_id)
            if eav_data:
                with st.spinner("Generating summary..."):
                    summary = generate_summary(st.session_state.patient_id, pd.DataFrame(eav_data, columns=['Attribute', 'Value']))
                
                st.subheader("Pathway Summary")
                st.text_area("Summary", summary, height=400)
                
                # Print and Save to PDF controls
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Print Summary"):
                        st.info("Please use your browser's print function to print this page.")
                with col2:
                    pdf_buffer = save_text_to_pdf(summary, f"pathway_summary_{st.session_state.patient_id}.pdf")
                    st.download_button(
                        label="Save to PDF",
                        data=pdf_buffer,
                        file_name=f"pathway_summary_{st.session_state.patient_id}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("No data available for the selected Pathway ID. Please select a valid Pathway ID first.")
        else:
            st.warning("Please select a Pathway ID before generating a summary.")

def manage_questions():
    st.title("Manage Questions")

    # Initialize session state
    if 'question_action' not in st.session_state:
        st.session_state.question_action = None

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Add Questions (SQL)"):
            st.session_state.question_action = "add"
            st.rerun()
    
    with col2:
        if st.button("Delete Questions (Graph)"):
            st.session_state.question_action = "delete"
            st.rerun()

    if st.session_state.question_action == "add":
        add_questions_sql()
    elif st.session_state.question_action == "delete":
        delete_graph_queries()

    # Reset action if queries were deleted
    if st.session_state.get('queries_deleted', False):
        st.session_state.queries_deleted = False
        st.session_state.question_action = "delete"  # Stay on the delete page
        st.rerun()

def add_questions_sql():
    st.subheader("Add New Question")
    new_question = st.text_area("Type your new question:", height=100)
    
    if st.button("Add Question"):
        if new_question:
            existing_questions = get_questions_from_db()
            matches = fuzzy_process.extract(new_question, existing_questions, limit=1)
            
            if matches and matches[0][1] >= 90:  # If similarity is 90% or higher
                st.warning(f"This question is similar to an existing question: '{matches[0][0]}'")
                if st.button("Add Anyway"):
                    add_question_to_db(new_question)
                    st.success("Question added successfully!")
            else:
                add_question_to_db(new_question)
                st.success("Question added successfully!")
        else:
            st.error("Please enter a question before adding.")

    if st.button("Back to Manage Questions"):
        st.session_state.question_action = None
        st.rerun()


def add_question():
    st.markdown('<p class="big-font">Add New Question</p>', unsafe_allow_html=True)
    new_question = st.text_area("Type your new question:", height=100)
    
    if st.button("Add Question"):
        if new_question:
            existing_questions = get_questions_from_db()
            matches = fuzzy_search(new_question, existing_questions, limit=1)
            
            if matches and matches[0][1] >= 90:  # If similarity is 90% or higher
                st.warning(f"This question is similar to an existing question: '{matches[0][0]}'")
                if st.button("Add Anyway"):
                    add_question_to_db(new_question)
                    st.success("Question added successfully!")
            else:
                add_question_to_db(new_question)
                st.success("Question added successfully!")
        else:
            st.error("Please enter a question before adding.")

def show_epoch_dashboard():
    def run_query(query):
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                result = session.run(query)
                return [record for record in result]

    # Queries to get the statistics
    queries = [
        ("Total Patients", "MATCH (p:Patient) RETURN count(p) AS count"),
        ("Unique Surgeons", "MATCH (st:SurgicalTeam) RETURN count(DISTINCT st.Clinician_txt) AS count"),
        ("Average Age", "MATCH (p:Patient) WHERE p.Age IS NOT NULL RETURN round(avg(toFloat(p.Age)), 1) AS avg_age"),
        ("Average BMI", "MATCH (p:Patient) WHERE p.BMI IS NOT NULL RETURN round(avg(toFloat(p.BMI)), 1) AS avg_bmi"),
        ("Gender Distribution", """
            MATCH (p:Patient)
            WHERE p.Gender_txt IS NOT NULL
            RETURN p.Gender_txt AS gender, count(*) AS count
            ORDER BY count DESC
        """),
        ("Average Tourniquet Time", """
            MATCH (proc:Procedure)
            WHERE proc.TourniquetTime IS NOT NULL
            RETURN round(avg(toFloat(proc.TourniquetTime)), 1) AS avg_time
        """),
        ("Most Used Implant", """
            MATCH (i:Implant)
            WHERE i.FemoralComponent_txt IS NOT NULL
            RETURN i.FemoralComponent_txt AS implant, count(*) AS count
            ORDER BY count DESC LIMIT 1
        """),
        ("Total Locations", "MATCH (l:Location) RETURN count(DISTINCT l) AS count"),
        ("Average Estimated Blood Loss", """
            MATCH (proc:Procedure)
            WHERE proc.EstimatedBloodLoss_txt IS NOT NULL
            RETURN round(avg(toFloat(replace(proc.EstimatedBloodLoss_txt, ' mL', ''))), 1) AS avg_blood_loss
        """)
    ]

    # Additional queries for surgeons, surgical centres, and implant types
    surgeon_query = """
    MATCH (st:SurgicalTeam)
    WHERE st.Clinician_txt IS NOT NULL
    RETURN DISTINCT st.Clinician_txt AS SurgeonName
    ORDER BY SurgeonName
    """

    centre_query = """
    MATCH (l:Location)<-[:PERFORMED_AT]-(p:Procedure)
    WHERE l.Location_txt IS NOT NULL
    RETURN DISTINCT l.Location_txt AS SurgicalCentre, COUNT(p) AS ProcedureCount
    ORDER BY ProcedureCount DESC
    """

    implant_query = """
    MATCH (i:Implant)
    WHERE i.FemoralComponent_txt IS NOT NULL
    RETURN DISTINCT i.FemoralComponent_txt AS ImplantType, COUNT(*) AS UsageCount
    ORDER BY UsageCount DESC
    """

    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Epoch Overview Dashboard</p>', unsafe_allow_html=True)

    # Create a 3x3 grid for the statistics
    cols = st.columns(3)
    for i, (title, query) in enumerate(queries):
        with cols[i % 3]:
            result = run_query(query)
            value = result[0][0] if result else "N/A"

            # For gender distribution, calculate percentage and include Male count
            if title == "Gender Distribution":
                total = sum(record[1] for record in result)
                female_count = next((r[1] for r in result if r[0] == 'Female'), 0)
                male_count = next((r[1] for r in result if r[0] == 'Male'), 0)
                value = f"Female: {female_count/total*100:.1f}%\nMale: {male_count/total*100:.1f}%\nMale Count: {male_count}"

            # For most used implant, include the name
            if title == "Most Used Implant":
                value = f"{value}: {result[0][1]}"

            st.markdown(f"""
                <div style="background-color: #4B0082; padding: 10px; border-radius: 10px; height: 200px; margin: 10px;">
                    <h3 style="color: #FFFFFF; font-size: 20px;">{title}</h3>
                    <p style="color: #FFFFFF; font-size: 18px; font-weight: bold; white-space: pre-line;">{value}</p>
                </div>
            """, unsafe_allow_html=True)

    # Function to display formatted list in a box
    def display_list_in_box(title, data, columns):
        items = "\n".join(f"<li>{' | '.join(str(item) for item in row)}</li>" for row in data)
        st.markdown(f"""
        <div style="background-color: #4B0082; padding: 10px; border-radius: 10px; height: 300px; margin: 10px; overflow-y: auto;">
            <h3 style="color: #FFFFFF; font-size: 20px;">{title}</h3>
            <ul style="color: #FFFFFF; font-size: 14px; list-style-type: none; padding-left: 0;">
                {items}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    # Display additional information
    cols = st.columns(3)
    with cols[0]:
        surgeon_data = run_query(surgeon_query)
        display_list_in_box("Surgeons", surgeon_data, ["Surgeon Name"])

    with cols[1]:
        centre_data = run_query(centre_query)
        display_list_in_box("Surgical Centres | Number of Procedures", centre_data, ["Surgical Centre", "Procedure Count"])

    with cols[2]:
        implant_data = run_query(implant_query)
        display_list_in_box("Implants Used", [(row[0], f"Count: {row[1]}") for row in implant_data], ["Implant Type", "Usage Count"])

def main_app_medacta():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.error("Please log in to access this page.")
        st.stop()

    set_page_background('white')

    # Custom CSS for styling
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .sidebar .sidebar-content .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .sidebar .sidebar-content h3 {
            color: #0e1117;
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            padding-bottom: 0.3rem;
            border-bottom: 1px solid #d1d5db;
        }
        .sidebar .sidebar-content .stSelectbox > label {
            font-size: 0.9rem;
            color: #4b5563;
        }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Medacta Research Project")

    # Initialize session state for last selected section
    if 'last_selected' not in st.session_state:
        st.session_state.last_selected = 'data_view'

    # Data View Section (Radio buttons with default selection)
    st.sidebar.markdown("### Data View")
    data_view = st.sidebar.radio("Data View", ["SQL Generator", "All Data", "Manage Questions"], index=0, key="data_view", label_visibility="collapsed", on_change=lambda: setattr(st.session_state, 'last_selected', 'data_view'))

    # Data Viz Section (Selectbox with initial empty option)
    st.sidebar.markdown("### Data Visualizations")
    data_viz = st.sidebar.selectbox("Data Visualizations", ["", "Epoch Dashboard", "Surgeries by Surgeon", "Distal Femoral Resection"], key="data_viz", label_visibility="collapsed", on_change=lambda: setattr(st.session_state, 'last_selected', 'data_viz') if st.session_state.data_viz else None)

    # AI Agent Examples Section (Selectbox with initial empty option)
    st.sidebar.markdown("### AI Agent Examples")
    agent_examples = st.sidebar.selectbox("AI Agent Examples", ["", "Clinical Question Agent", "EAV Searcher", "Graph Query"], key="agent_examples", label_visibility="collapsed", on_change=lambda: setattr(st.session_state, 'last_selected', 'agent_examples') if st.session_state.agent_examples else None)

    # Determine which page to show based on the last selected section
    if st.session_state.last_selected == 'data_view':
        if data_view == "SQL Generator":
            show_prompt_screen()
        elif data_view == "All Data":
            show_all_data()
        elif data_view == "Manage Questions":
            manage_questions()
    elif st.session_state.last_selected == 'data_viz' and data_viz:
        if data_viz == "Epoch Dashboard":
            show_epoch_dashboard()
        elif data_viz == "Surgeries by Surgeon":
            show_surgeries_by_surgeon()
        elif data_viz == "Distal Femoral Resection":
            show_distal_femoral_resection_dashboard()
    elif st.session_state.last_selected == 'agent_examples' and agent_examples:
        if agent_examples == "Clinical Question Agent":
            main_app()
        elif agent_examples == "EAV Searcher":
            eav_searcher()
        elif agent_examples == "Graph Query":
            graph_qa_page()
    else:
        # Default view (first option of the first menu)
        show_prompt_screen()


if __name__ == "__main__":
    main_app_medacta()