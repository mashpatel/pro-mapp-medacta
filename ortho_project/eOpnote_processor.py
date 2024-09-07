import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import PyPDF2
import json
import re

# Load environment variables
load_dotenv()

# Get the API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is loaded
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Initialize the Google Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=google_api_key,
    temperature=0.5
)

prompt_template = """
You are a clinical expert in knee surgery and experienced in extracting medical information from operative notes and mapping it to standardized forms.

Given the following operative note text, extract relevant information and map it to the corresponding fields in the initial sections of the National Joint Registry (NJR) form. Focus on patient details, operation details, and surgeon information.

Operative Note Text:
{operative_note_text}

IMPORTANT: Instructions for Extraction and Output:
1. Extract information ONLY for the sections and fields specified below.
2. If information for a field is not available or unclear, use "NO" as the value.
3. Use your expert knowledge to infer information where appropriate, but do not guess if uncertain.
4. Ensure your response is a complete and valid JSON object.
5. Do not include any explanatory text or markdown formatting in your response.

Output the result in the following format:

{{
  "form_title": "MDS VERSION 7.0 Knee Operation Form: MDSv7.0 K1 v2.0 K1 Knee Primary",
  "sections": [
    {{
      "section_title": "PATIENT DETAILS",
      "fields": [
        {{
          "field_name": "NJR Patient Consent Obtained",
          "options": ["Yes", "No", "Not Recorded"],
          "value": ""
        }},
        {{
          "field_name": "Body Mass Index",
          "value": ""
        }},
        {{
          "field_name": "Height (IN M)",
          "value": ""
        }},
        {{
          "field_name": "Weight (IN KG)",
          "value": ""
        }}
      ]
    }},
    {{
      "section_title": "PATIENT IDENTIFIERS",
      "fields": [
        {{
          "field_name": "Forename(s)",
          "value": ""
        }},
        {{
          "field_name": "Surname",
          "value": ""
        }},
        {{
          "field_name": "Gender",
          "options": ["Male", "Female", "Not Known", "Not Specified"],
          "value": ""
        }},
        {{
          "field_name": "Date of Birth",
          "format": "DD/MM/YYYY",
          "value": ""
        }},
        {{
          "field_name": "NHS Number or National Patient Identifier (if available)",
          "value": ""
        }},
        {{
          "field_name": "Patient Hospital ID",
          "value": ""
        }}
      ]
    }},
    {{
      "section_title": "OPERATION DETAILS",
      "fields": [
        {{
          "field_name": "Hospital",
          "value": ""
        }},
        {{
          "field_name": "Operation Date",
          "format": "DD/MM/YYYY",
          "value": ""
        }},
        {{
          "field_name": "Anaesthetic Types",
          "options": ["General", "Regional - Epidural", "Regional – Nerve Block", "Regional – Spinal (Intrathecal)"],
          "value": []
        }},
        {{
          "field_name": "Patient ASA Grade",
          "options": ["1", "2", "3", "4", "5"],
          "value": ""
        }},
        {{
          "field_name": "Operation Funding",
          "options": ["NHS", "Independent"],
          "value": ""
        }}
      ]
    }},
    {{
      "section_title": "SURGEON DETAILS",
      "fields": [
        {{
          "field_name": "Consultant in Charge",
          "value": ""
        }},
        {{
          "field_name": "Operating Surgeon",
          "value": ""
        }},
        {{
          "field_name": "Operating Surgeon Grade",
          "options": ["Consultant", "SPR/ST3-8", "F1-ST2", "Specialty Doctor/SAS", "Other"],
          "value": ""
        }},
        {{
          "field_name": "First Assistant Grade",
          "options": ["Consultant", "Other"],
          "value": ""
        }}
      ]
    }}
  ]
}}

FINAL INSTRUCTION: Before submitting your response, verify that you have produced a complete, valid JSON object with all sections properly closed. Ensure that no fields are omitted and that the structure exactly matches the provided format. For fields with options, ensure the value is either a single selected option (as a string) or an array of selected options, as appropriate. For fields without options, provide the extracted information as a string or "NO" if the information is not available.
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["operative_note_text"])
chain = LLMChain(llm=llm, prompt=prompt)

def process_ai_response(raw_response):
    """
    Process the AI response, handling the new detailed JSON structure.
    """
    try:
        output = json.loads(raw_response)
        
        # Extract the form title
        form_title = output.get('form_title', '')
        
        # Process each section
        processed_sections = []
        for section in output.get('sections', []):
            processed_fields = []
            for field in section.get('fields', []):
                processed_fields.append({
                    'name': field.get('field_name', ''),
                    'value': field.get('value', '')
                })
            
            processed_sections.append({
                'title': section.get('section_title', ''),
                'fields': processed_fields
            })
        
        return {
            'form_title': form_title,
            'sections': processed_sections
        }
    except json.JSONDecodeError as e:
        st.error(f"Error parsing AI response: {e}")
        st.text("Raw response causing the error:")
        st.text(raw_response)
        return {}

def clean_json_response(response):
    """
    Remove code block markers and any leading/trailing whitespace from the response.
    """
    # Remove ```json at the start and ``` at the end
    response = re.sub(r'^```json\s*', '', response)
    response = re.sub(r'\s*```$', '', response)
    return response.strip()

def process_operative_note(uploaded_file):
    """
    Processes the uploaded operative note PDF and extracts NJR data using Google Gemini.
    """
    # Extract text from the PDF
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    operative_note_text = ""
    for page in pdf_reader.pages:
        operative_note_text += page.extract_text()

    # Use LLMChain to process the operative note
    response = chain.run(operative_note_text=operative_note_text)
    
    # Clean the response
    cleaned_response = clean_json_response(response)

    return operative_note_text, cleaned_response

# Streamlit UI
st.title("NJR Form Mapper")

uploaded_file = st.file_uploader("Upload Operative Note (PDF)", type="pdf")

# In your Streamlit app's main logic:
if st.button("Process Operative Note"):
    if uploaded_file is not None:
        with st.spinner("Processing operative note..."):
            extracted_text, raw_response = process_operative_note(uploaded_file)

        with st.expander("Extracted Text from PDF"):
            st.text(extracted_text)

        with st.expander("Raw AI Response"):
            st.text(raw_response)

        processed_data = process_ai_response(raw_response)

        if processed_data:
            st.subheader(processed_data['form_title'])
            for section in processed_data['sections']:
                with st.expander(section['title']):
                    for field in section['fields']:
                        st.text(f"{field['name']}: {field['value']}")
        else:
            st.error("Failed to process the AI response.")

    else:
        st.warning("Please upload an operative note first.")