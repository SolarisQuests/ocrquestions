from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
from dotenv import load_dotenv
import os
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
mongo_url = os.getenv("MONGODB_URI")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


client = AsyncIOMotorClient(mongo_url)
db = client["Documenttask"]
images_collection = db["imagesdemo"]
questions_collection = db["Common_questions"]

async def get_openai_response(prompt):
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Always provide concise answers without restating the question. Date as YYYYMMDD format. Names as Last Name,First Name Middle Name format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500
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

    print(f"Processed document: {document['_id']} with new question: {question}")
    return response

 async def process_all_documents(question: str):
    cursor = images_collection.find()
    documents = await cursor.to_list(length=None)
    if not documents:
        raise HTTPException(status_code=404, detail="No documents found in the collection")
    responses = ''
    for document in documents:
        document_text = "\n".join([item for sublist in document.get("json_data", []) for item in sublist.values()])
        # llm
        prompt = f"Document text:\n{document_text}\n\nQuestion: {question}\n\nAnswer (provide a precise and concise response without restating the question):"
        response = await get_openai_response(prompt)
        responses+=response
    await questions_collection.insert_one({
        "question": question,
        "responses": responses,
        "question_asked_time": datetime.utcnow()
    })
    return responses

@app.get("/")
async def read_root():
    return {"status": "ok"}

@app.post("/ask_question")
async def ask_question(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        print(f"Failed to parse JSON body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    doc_id = body.get("id")
    question = body.get("question")
    
    if not doc_id or not question:
        raise HTTPException(status_code=400, detail="Both 'id' and 'question' fields are required")

    response = await process_specific_question(doc_id, question)
    return JSONResponse(content={"message": "Question answered successfully", "response": response}, status_code=200)

@app.post("/ask_all_documents")
async def ask_all_documents(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        print(f"Failed to parse JSON body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    question = body.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="'question' field is required")

    responses = await process_all_documents(question)
    return JSONResponse(content={"responses": responses}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
