import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def run_query(query):
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

# Streamlit app
st.set_page_config(layout="wide", page_title="Epoch Overview Dashboard")

st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
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
            <div style="background-color: #4B0082; padding: 20px; border-radius: 10px; height: 200px; margin: 10px;">
                <h3 style="color: #FFFFFF; font-size: 24px;">{title}</h3>
                <p style="color: #FFFFFF; font-size: 36px; font-weight: bold; white-space: pre-line;">{value}</p>
            </div>
        """, unsafe_allow_html=True)

# Function to display formatted list in a box
def display_list_in_box(title, data, columns):
    items = "\n".join(f"<li>{' | '.join(str(item) for item in row)}</li>" for row in data)
    st.markdown(f"""
        <div style="background-color: #4B0082; padding: 20px; border-radius: 10px; height: 300px; margin: 10px; overflow-y: auto;">
            <h3 style="color: #FFFFFF; font-size: 24px;">{title}</h3>
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

# Close the Neo4j connection
driver.close()
