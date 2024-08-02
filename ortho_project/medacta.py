import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import streamlit as st
import sqlite3
import pandas as pd
import traceback
import plotly.express as px
from datetime import datetime, time

anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Function to get data from SQLite database
def get_data(query, params=None):
    conn = sqlite3.connect('medacta.db')
    if params:
        df = pd.read_sql_query(query, conn, params=params)
    else:
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def generate_sql_query(user_input, constraints, explain=False):
    prompt = f"""You are an AI assistant specialized in translating natural language queries into SQL for a medical database called medacta.db. Your task is to generate SQL queries based on user input while adhering to specific constraints.

Database Schema:
{{database_schema}}

User Input:
{user_input}

Constraints and Instructions:
{constraints}

General Rules:
1. Only use fields that exist in the provided database schema.
2. If the user input doesn't translate to a sensible query, return "No query formed".
3. Ensure the query is syntactically correct for SQLite.
4. Do not use any fields or tables not explicitly defined in the schema.
5. Always consider the given constraints when forming the query.

Your response should be ONLY the SQL query, without any additional explanation or context, unless explicitly requested.

If you cannot form a valid query based on the input and constraints, respond with "No query formed".

SQL Query:
"""

    response = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    sql_query = response.content[0].text.strip()

    if explain and sql_query != "No query formed":
        explanation_prompt = f"{prompt}\n\nThe generated SQL query is:\n{sql_query}\n\nPlease explain this SQL query and how it relates to the user's input:"
        explanation_response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": explanation_prompt}
            ]
        )
        return sql_query, explanation_response.content[0].text.strip()
    else:
        return sql_query

# Streamlit app
def main():
    # Sidebar for navigation
    st.sidebar.title("Analysis")
    page = st.sidebar.radio("Go to", ["Prompt", "", "Surgeries by Surgeon", "Medial Condyle", "All Data","EAV Searcher"])

    if page == "Prompt":
        show_prompt_screen()
    elif page == "Surgeries by Surgeon":
        show_surgeries_by_surgeon()
    elif page == "Medial Condyle":
        show_medial_condyle_analysis()
    elif page == "All Data":
        show_all_data()
    elif page == "EAV Searcher":
        eav_searcher()

def execute_sql_query(sql_query):
    conn = sqlite3.connect('medacta.db')
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df

def get_eav_data(patient_pid):
    conn = sqlite3.connect('Medacta.db')
    cursor = conn.cursor()
    cursor.execute("SELECT Attribute, Value FROM Surgical_Data_EAV WHERE EntityID = ?", (patient_pid,))
    eav_data = cursor.fetchall()
    conn.close()
    return eav_data

def eav_searcher():
    st.header("EAV Searcher")

    # Initialize session state for patient_id if it doesn't exist
    if 'patient_id' not in st.session_state:
        st.session_state.patient_id = ""

    # Add a reset button
    if st.button("Reset Search", key="reset_button", help="Click to reset the search"):
        st.session_state.patient_id = ""
        st.experimental_rerun()

    patient_id = st.text_input("Enter Patient ID:", value=st.session_state.patient_id)
    st.session_state.patient_id = patient_id
    
    if patient_id:
        eav_data = get_eav_data(patient_id)
        if eav_data:
            st.subheader(f"Results for Pathway ID: {patient_id}")
            
            # Convert the data to a DataFrame
            df = pd.DataFrame(eav_data, columns=['Attribute', 'Value'])
            
            # Display attribute-value pairs in boxes
            for _, row in df.iterrows():
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 5px; border-radius: 5px;">
                    <strong>{row['Attribute']}:</strong> {row['Value']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"No data found for the given Pathway ID: {patient_id}")
            st.info("Please check the ID and try again. If the problem persists, the table structure might be different from what's expected.")

# CSS for compact layout and reset button
st.markdown("""
<style>
    .stButton>button {
        color: white;
        background-color: #0000FF;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0000CC;
    }
    .element-container {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def show_prompt_screen():
    st.header("SQL Query Generator")
    
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if 'constraints' not in st.session_state:
        st.session_state.constraints = "Always use the field CompleteDate when compiling your query.\nNever use .txt fields in your queries."
    
    user_input = st.text_area("Enter your query in natural language:", height=150, value=st.session_state.user_input)
    constraints = st.text_area("Enter any constraints or instructions:", height=100, value=st.session_state.constraints)
    explain = st.checkbox("Explain the generated query")
    
    if st.button("Generate SQL"):
        if user_input:
            with st.spinner("Generating SQL query..."):
                result = generate_sql_query(user_input, constraints, explain)
                if isinstance(result, tuple):
                    sql_query, explanation = result
                    st.code(sql_query, language="sql")
                    st.subheader("Explanation:")
                    st.write(explanation)
                else:
                    sql_query = result
                    st.code(sql_query, language="sql")
                
                # Execute the SQL query and display results
                if sql_query != "No query formed":
                    try:
                        df = execute_sql_query(sql_query)
                        st.write("Query Results:")
                        st.dataframe(df)
                    except Exception as e:
                        st.error(f"Error executing query: {str(e)}")
        else:
            st.warning("Please enter a query before submitting.")
    
    # Clear button to reset the inputs
    if st.button("Clear"):
        st.session_state.user_input = ""
        st.session_state.constraints = "Always use the field CompleteDate when compiling your query.\nNever use .txt fields in your queries."
        st.experimental_rerun()

    # Store the current values in session state
    st.session_state.user_input = user_input
    st.session_state.constraints = constraints

def show_surgeries_by_surgeon():
    st.header("Number of Surgeries by Surgeon")

    # Date range filter
    min_date = pd.to_datetime(get_data("SELECT MIN(CompleteDate) FROM Surgical_Data").iloc[0, 0])
    max_date = pd.to_datetime(get_data("SELECT MAX(CompleteDate) FROM Surgical_Data").iloc[0, 0])

    st.write(f"Date range in database: {min_date.date()} to {max_date.date()}")

    # Two-way date range slider
    date_range = st.slider(
        "Select date range",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date()),
        format="YYYY-MM-DD"
    )
    start_date, end_date = date_range

    # Query for surgeries by surgeon with date filter
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

    # Display the query results
    st.write("Number of surgeries performed by each surgeon:")
    st.dataframe(surgeon_data)

    # Create a bar chart for surgeries by surgeon
    fig = px.bar(surgeon_data, x='Surgeon', y='SurgeryCount', 
                 title='Number of Surgeries by Surgeon',
                 labels={'SurgeryCount': 'Number of Surgeries'})
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig)

def show_medial_condyle_analysis():
    st.header("Medial Condyle Initial Thickness Analysis")

    # Date range filter
    min_date = pd.to_datetime(get_data("SELECT MIN(CompleteDate) FROM Surgical_Data").iloc[0, 0])
    max_date = pd.to_datetime(get_data("SELECT MAX(CompleteDate) FROM Surgical_Data").iloc[0, 0])

    st.write(f"Date range in database: {min_date.date()} to {max_date.date()}")

    # Two-way date range slider
    date_range = st.slider(
        "Select date range",
        min_value=min_date.date(),
        max_value=max_date.date(),
        value=(min_date.date(), max_date.date()),
        format="YYYY-MM-DD"
    )
    start_date, end_date = date_range

    # Create a toggle for Worn, Unworn, and All
    condyle_status = st.radio("Select Medial Condyle Status", ["All", "Worn", "Unworn"])

    # Construct the WHERE clause based on the selection
    where_clause = "WHERE date(CompleteDate) BETWEEN ? AND ?"
    params = [start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
    
    if condyle_status == "Worn":
        where_clause += " AND DFRMedialCondyleStatus = 1"
    elif condyle_status == "Unworn":
        where_clause += " AND DFRMedialCondyleStatus = 2"

    # Query for Medial Condyle Initial Thickness
    medial_condyle_query = f"""
    SELECT 
        DFRMedialCondyleInitialThickness AS InitialThickness,
        COUNT(*) AS PatientCount,
        DFRMedialCondyleStatus_txt AS Status
    FROM 
        Surgical_Data
    {where_clause}
    GROUP BY 
        DFRMedialCondyleInitialThickness, DFRMedialCondyleStatus_txt
    ORDER BY 
        DFRMedialCondyleInitialThickness
    """
    
    medial_condyle_data = get_data(medial_condyle_query, params=params)

    # Display the query results
    st.write(f"Medial Condyle Initial Thickness Data ({condyle_status}):")
    st.dataframe(medial_condyle_data)

    # Create a bar chart for Medial Condyle Initial Thickness
    fig = px.bar(medial_condyle_data, x='InitialThickness', y='PatientCount', 
                 color='Status',
                 title=f'Medial Condyle Initial Thickness Distribution ({condyle_status})',
                 labels={'InitialThickness': 'Initial Thickness (mm)', 'PatientCount': 'Number of Patients'})
    fig.update_layout(barmode='group')  # This will group the bars by Status
    st.plotly_chart(fig)

def show_all_data():
    st.header("All Surgical Data")
    
    # Fetch all data
    all_data = get_data("SELECT * FROM Surgical_Data")

    # Display all data
    st.write("Here's your complete dataset:")
    st.dataframe(all_data)

database_schema = ""

    

if __name__ == "__main__":
    # In your main() function or wherever you call show_prompt_screen()
    if 'prompt' not in st.session_state:
        st.session_state.prompt = ""
    main()