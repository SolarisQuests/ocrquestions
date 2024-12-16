from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
from dotenv import load_dotenv
import os
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime
import logging
from pydantic import BaseModel
from functools import lru_cache
from pydantic_settings import BaseSettings
from fastapi.middleware import Middleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
mongo_url = os.getenv("MONGODB_URI")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Document QA API",
    description="API for querying documents using OpenAI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

client = AsyncIOMotorClient(mongo_url)
db = client["Documenttask"]
images_collection = db["imagesdemo"]
questions_collection = db["Common_questions"]

class Settings(BaseSettings):
    openai_api_key: str
    mongodb_uri: str
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 1500

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()

async def get_openai_response(prompt):
    response = await openai.ChatCompletion.acreate(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Always provide concise answers without restating the question. Date as YYYYMMDD format. Names as Last Name,First Name Middle Name format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=settings.max_tokens
    )
    return response.choices[0].message['content'].strip()

async def process_specific_question(doc_id: str, question: str):
    try:
        object_id = ObjectId(doc_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ObjectId: {e}")

    document = await images_collection.find_one({"_id": object_id})
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # combine the json_data into a single text
    document_text = "\n".join([item for sublist in document.get("json_data", []) for item in sublist.values()])

    # generate the OpenAI response
    prompt = f"Document text:\n{document_text}\n\nQuestion: {question}\n\nAnswer (provide a precise and concise response without restating the question):"
    response = await get_openai_response(prompt)

    # retrieve existing questions and results
    prompts_questions = document.get("prompts_questions", [])
    prompts_results = document.get("prompts_results", [])

    # aappend the new question and response
    prompts_questions.append(question)
    prompts_results.append(response)

    # update the document with the new questions and responses
    await images_collection.update_one(
        {"_id": document["_id"]},
        {"$set": {
            "prompts_questions": prompts_questions,
            "prompts_results": prompts_results
        }}
    )

    logger.info(f"Processed document: {document['_id']} with new question: {question}")
    return response

async def process_all_documents(question: str):
    # Get all documents
    cursor = images_collection.find()
    documents = await cursor.to_list(length=None)
    
    if not documents:
        raise HTTPException(status_code=404, detail="No documents found in the collection")

    responses = []
    document_sources = []  # To track which document provided which response
    
    for document in documents:
        document_text = "\n".join([item for sublist in document.get("json_data", []) for item in sublist.values()])
        
        # Generate the OpenAI response for each document
        prompt = f"""Document text:
{document_text}

Question: {question}

Instructions:
1. Answer based only on the information in this document
2. If the document doesn't contain relevant information, respond with "No relevant information found in this document"
3. Be precise and concise
4. Do not restate the question

Answer:"""
        
        response = await get_openai_response(prompt)
        
        # Only include responses that found relevant information
        if "No relevant information found in this document" not in response:
            responses.append(response)
            document_sources.append(str(document['_id']))
        
    # Save in db with document sources
    await questions_collection.insert_one({
        "question": question,
        "responses": responses,
        "document_sources": document_sources,
        "question_asked_time": datetime.utcnow(),
        "is_global_question": True,
        "documents_analyzed": len(documents),
        "documents_with_answers": len(responses)
    })
    
    return {
        "responses": responses,
        "document_sources": document_sources,
        "total_documents": len(documents),
        "documents_with_answers": len(responses)
    }

@app.get("/")
async def read_root():
    return {"status": "ok"}

class QuestionRequest(BaseModel):
    id: str
    question: str

class AllDocumentsRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    message: str
    response: str

class AllDocumentsResponse(BaseModel):
    responses: List[str]

class GlobalChatResponse(BaseModel):
    responses: List[str]
    message: str
    documents_analyzed: int

@app.post("/ask_question", response_model=QuestionResponse)
@limiter.limit("5/minute")  # Limit to 5 requests per minute
async def ask_question(request: QuestionRequest):
    response = await process_specific_question(request.id, request.question)
    return JSONResponse(content={"message": "Question answered successfully", "response": response}, status_code=200)

@app.post("/ask_all_documents")
async def ask_all_documents(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    question = body.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="'question' field is required")

    result = await process_all_documents(question)
    return JSONResponse(content={
        "responses": result["responses"],
        "document_sources": result["document_sources"],
        "total_documents_analyzed": result["total_documents"],
        "documents_with_answers": result["documents_with_answers"]
    }, status_code=200)

@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(settings.mongodb_uri)
    app.mongodb = app.mongodb_client["Documenttask"]

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
