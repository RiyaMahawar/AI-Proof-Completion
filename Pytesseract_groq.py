import os
import io
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from sqlalchemy import create_engine, Column, Integer, String, Date, MetaData, Table
from sqlalchemy.orm import sessionmaker

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize FastAPI app
app = FastAPI()

# Initialize Groq LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key=GROQ_API_KEY
)
output_parser = StrOutputParser()

# Prompt template for Aadhaar
aadhaar_prompt = PromptTemplate.from_template("""
You are an intelligent document parser.

Here is the OCR extracted text from an Aadhaar card:

{document_text}

Extract the following fields:
- Full Name
- Date of Birth (format: YYYY-MM-DD)
- Aadhaar Number (ignore spaces if present) (12 digit no. present at the bottom of the card)

Respond only with JSON:
{{
    "name": "...",
    "dob": "...",
    "aadhar": "..."
}}
""")

# Prompt template for PAN (PAN number only)
pan_prompt = PromptTemplate.from_template("""
You are an intelligent document parser.

Here is the OCR extracted text from an Indian PAN card:

{document_text}

Extract only the PAN Number (ignore everything else). The PAN number is a 10-character alphanumeric code (5 uppercase letters, 4 digits, 1 uppercase letter).

Respond only with JSON:
{{
    "pan": "..."
}}
""")

# Database setup
engine = create_engine(DATABASE_URL)
metadata = MetaData()
documents_table = Table(
    "documents",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String),
    Column("dob", Date),
    Column("aadhar", String),
    Column("pan", String),
)
metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def clean_aadhar_number(aadhar: str) -> str:
    return aadhar.replace(" ", "").strip()

@app.post("/upload-documents/")
async def upload_documents(aadhaar_file: UploadFile = File(...), pan_file: UploadFile = File(...)):
    try:
        # === Aadhaar OCR and LLM Parsing ===
        aadhaar_bytes = await aadhaar_file.read()
        aadhaar_image = Image.open(io.BytesIO(aadhaar_bytes))
        aadhaar_text = pytesseract.image_to_string(aadhaar_image)

        aadhaar_chain = aadhaar_prompt | llm | output_parser
        aadhaar_result_raw = aadhaar_chain.invoke({"document_text": aadhaar_text})
        aadhaar_result_cleaned = aadhaar_result_raw.strip().strip("`").strip()
        try:
            aadhaar_data = json.loads(aadhaar_result_cleaned)
            aadhaar_data["aadhar"] = clean_aadhar_number(aadhaar_data["aadhar"])
        except Exception:
            return JSONResponse(content={
                "error": "Failed to parse Aadhaar output",
                "raw": aadhaar_result_raw
            }, status_code=400)

        # === PAN OCR and LLM Parsing (PAN number only) ===
        pan_bytes = await pan_file.read()
        pan_image = Image.open(io.BytesIO(pan_bytes))
        pan_text = pytesseract.image_to_string(pan_image)

        pan_chain = pan_prompt | llm | output_parser
        pan_result_raw = pan_chain.invoke({"document_text": pan_text})
        pan_result_cleaned = pan_result_raw.strip().strip("`").strip()
        try:
            pan_data = json.loads(pan_result_cleaned)
        except Exception:
            return JSONResponse(content={
                "error": "Failed to parse PAN output",
                "raw": pan_result_raw
            }, status_code=400)

        # === Insert into DB ===
        session = SessionLocal()
        insert_stmt = documents_table.insert().values(
            name=aadhaar_data["name"].strip(),
            dob=aadhaar_data["dob"],
            aadhar=aadhaar_data["aadhar"],
            pan=pan_data["pan"].strip()
        )
        session.execute(insert_stmt)
        session.commit()

        return JSONResponse(content={
            "message": "Document data inserted successfully",
            "data": {
                "name": aadhaar_data["name"],
                "dob": aadhaar_data["dob"],
                "aadhar": aadhaar_data["aadhar"],
                "pan": pan_data["pan"]
            }
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
