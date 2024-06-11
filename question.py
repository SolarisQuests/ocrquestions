# from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import openai
# from dotenv import load_dotenv
# import os
# from pymongo import MongoClient
# from bson import ObjectId

# # Load environment variables
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# mongo_url = os.getenv("MONGODB_API_KEY")

# # Connect to MongoDB
# client = MongoClient(mongo_url)
# db = client["Documenttask"]
# images_collection = db["images"]

# # Function to get response from OpenAI
# def get_openai_response(prompt):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant. Always provide concise answers without restating the question. Date as YYYYMMDD format. Names as Last Name, First Name Middle Name format."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=1500
#     )
#     return response.choices[0].message['content'].strip()

# # Function to process a specific document by ID and question
# def process_specific_question(doc_id, question):
#     document = images_collection.find_one({"_id": ObjectId(doc_id)})
        
#     if not document:
#         raise HTTPException(status_code=404, detail="Document not found")

#     # Combine the json_data into a single text
#     document_text = "\n".join(document.get("json_data", []))
    
#     # Generate the OpenAI response
#     prompt = f"Document text:\n{document_text}\n\nQuestion: {question}\n\nAnswer (provide a precise and concise response without restating the question):"
#     response = get_openai_response(prompt)

#     # Retrieve existing questions and results
#     prompts_questions = document.get("prompts_questions", [])
#     prompts_results = document.get("prompts_results", [])

#     # Append the new question and response
#     prompts_questions.append(question)
#     prompts_results.append(response)

#     # Update the document with the new questions and responses
#     images_collection.update_one(
#         {"_id": document["_id"]},
#         {"$set": {
#             "prompts_questions": prompts_questions,
#             "prompts_results": prompts_results
#         }}
#     )
#     print(f"Processed document: {document['_id']} with new question: {question}")

#     return response

# # Initialize FastAPI app
# app = FastAPI()

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow any origin
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/ask_question")
# async def ask_question(request: Request):
#     try:
#         body = await request.json()
#     except Exception as e:
#         print(f"Failed to parse JSON body: {e}")
#         raise HTTPException(status_code=400, detail="Invalid JSON payload")

#     doc_id = body.get("id")
#     question = body.get("question")
#     if not doc_id or not question:
#         raise HTTPException(status_code=400, detail="Both 'id' and 'question' fields are required")

#     response = process_specific_question(doc_id, question)
#     return JSONResponse(content={"message": "Question answered successfully", "response": response}, status_code=200)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from bson import ObjectId
# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
mongo_url = os.getenv("MONGODB_API_KEY")
# Connect to MongoDB
client = MongoClient(mongo_url)
db = client["Documenttask"]
images_collection = db["images"]
# Function to get response from OpenAI
def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Always provide concise answers without restating the question. Date as YYYYMMDD format. Names as Last Name, First Name Middle Name format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500
    )
    return response.choices[0].message['content'].strip()
# Function to process a specific document by ID and question
def process_specific_question(doc_id, question):
    # print(doc_id)
    # document = images_collection.find_one({"_id": ObjectId(f"{doc_id}")})
    try:
        object_id = ObjectId(doc_id)
        print(object_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ObjectId: {e}")
    document = images_collection.find_one({"_id": object_id})
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    # Combine the json_data into a single text
    document_text = "\n".join([item for sublist in document.get("json_data", []) for item in sublist.values()])
    # Generate the OpenAI response
    prompt = f"Document text:\n{document_text}\n\nQuestion: {question}\n\nAnswer (provide a precise and concise response without restating the question):"
    response = get_openai_response(prompt)
    # Retrieve existing questions and results
    prompts_questions = document.get("prompts_questions", [])
    prompts_results = document.get("prompts_results", [])
    # Append the new question and response
    prompts_questions.append(question)
    prompts_results.append(response)
    # Update the document with the new questions and responses
    images_collection.update_one(
        {"_id": document["_id"]},
        {"$set": {
            "prompts_questions": prompts_questions,
            "prompts_results": prompts_results
        }}
    )
    print(f"Processed document: {document['_id']} with new question: {question}")
    return response
# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    print(doc_id)
    question = body.get("question")
    if not doc_id or not question:
        raise HTTPException(status_code=400, detail="Both 'id' and 'question' fields are required")
    response = process_specific_question(doc_id, question)
    return JSONResponse(content={"message": "Question answered successfully", "response": response}, status_code=200)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
