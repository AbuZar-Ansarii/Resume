from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Annotated
import streamlit as st
import PyPDF2
import io
import json

class Resume(BaseModel):
    name: str
    category: str
    marks: Annotated[float, Field(description="give marks based on skill ,out of 100")]
    skills: str
    summary: str

def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file."""
    try:
        pdf_file = io.BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except PyPDF2.errors.PdfReadError:
        return "Error: Unable to read PDF. The file may be corrupted or encrypted."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

st.title("RESUME CHECKER")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
structured_model = model.with_structured_output(Resume)

if st.button("Check Resume"):
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        extracted_text = extract_text_from_pdf(file_bytes)

        if isinstance(extracted_text, str) and not extracted_text.startswith("Error"):
            try:
                result = structured_model.invoke(extracted_text)
                st.write("Marks Out Of 100")
                st.write(result)

                # Add Download Button
                result_json = result.model_dump_json() # Convert Pydantic object to JSON
                st.download_button(
                    label="Download Result as JSON",
                    data=result_json,
                    file_name="resume_analysis.json",
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"Error processing resume: {e}")

        else:
            st.error(extracted_text)
    else:
        st.warning("Please upload a PDF file.")

