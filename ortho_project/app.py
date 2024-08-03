import streamlit as st
import os
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from anthropic import Anthropic
import sqlite3
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from typing import List
from datetime import datetime
from fuzzywuzzy import process

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

# Initialize Anthropic client
anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
    data_view = st.sidebar.radio("Data View", ["SQL Generator", "All Data", "Add Question"], index=0, key="data_view", label_visibility="collapsed", on_change=lambda: setattr(st.session_state, 'last_selected', 'data_view'))

    # Data Viz Section (Selectbox with initial empty option)
    st.sidebar.markdown("### Data Visualizations")
    data_viz = st.sidebar.selectbox("Data Visualizations", ["", "Surgeries by Surgeon", "Distal Femoral Resection"], key="data_viz", label_visibility="collapsed", on_change=lambda: setattr(st.session_state, 'last_selected', 'data_viz') if st.session_state.data_viz else None)

    # AI Agent Examples Section (Selectbox with initial empty option)
    st.sidebar.markdown("### AI Agent Examples")
    agent_examples = st.sidebar.selectbox("AI Agent Examples", ["", "Clinical Question Agent", "EAV Searcher"], key="agent_examples", label_visibility="collapsed", on_change=lambda: setattr(st.session_state, 'last_selected', 'agent_examples') if st.session_state.agent_examples else None)

    # Determine which page to show based on the last selected section
    if st.session_state.last_selected == 'data_view':
        if data_view == "SQL Generator":
            show_prompt_screen()
        elif data_view == "All Data":
            show_all_data()
        elif data_view == "Add Question":
            add_question()
    elif st.session_state.last_selected == 'data_viz' and data_viz:
        if data_viz == "Surgeries by Surgeon":
            show_surgeries_by_surgeon()
        elif data_viz == "Distal Femoral Resection":
            show_distal_femoral_resection_dashboard()
    elif st.session_state.last_selected == 'agent_examples' and agent_examples:
        if agent_examples == "Clinical Question Agent":
            main_app()
        elif agent_examples == "EAV Searcher":
            eav_searcher()
    else:
        # Default view (first option of the first menu)
        show_prompt_screen()


if __name__ == "__main__":
    main_app_medacta()