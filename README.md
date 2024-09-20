# Flask Chatbot Application

## Overview
This Flask application implements a context-aware chatbot integrated with a knowledge base using MongoDB. It leverages various models from Hugging Face for natural language processing and user intent recognition.

## Features
User Authentication: Session management to track user interactions.
Conversation Management: Store and retrieve conversations in a MongoDB database.
Intent Recognition: Classifies user intents using pretrained models.
Fallback Responses: Provides helpful responses when user input is not understood.
Knowledge Base: Maintains a knowledge base that can be queried for information.
File Upload Support: Extracts text from images and PDFs using Tesseract OCR and other libraries.

## Requirements
Python 3.8+
Flask
Flask-PyMongo
PyTorch
Transformers
Sentence Transformers
requests
pytesseract
MongoDB
Installation


Clone the repository:
`git clone https://github.com/Pj-develop/Hackrx5`
`cd yourrepository`
Set up a virtual environment (optional but recommended):
`python -m venv venv`
`source venv/bin/activate`  
# On Windows use `venv\Scripts\activate`

Install required packages:
pip install -r requirements.txt

Set environment variables for Hugging Face API key and MongoDB URI:
export hugging_face_key='YOUR_HUGGING_FACE_KEY'
export MONGODB_URI='YOUR_MONGODB_URI'

## Run the application:

python app.py

## API Endpoints
GET /: Check if the server is running.
GET /test: Test endpoint for server status.
POST /chat: Send user input to the chatbot and receive a response.
Request Body:
json
Copy code
{
    "user_input": "Hello!"
}
POST /conversations: Create a new conversation entry.
GET /conversations/getsession: Retrieve conversations for the current user session.
PUT /conversations/update/<conversation_id>: Update a specific conversation.
DELETE /conversations/delete/<conversation_id>: Delete a specific conversation.
GET /knowledge_base: Retrieve all entries in the knowledge base.
GET /session: Retrieve current session information.
POST /logout: Log out the user and clear session data.
Usage
Interact with the chatbot by sending POST requests to the /chat endpoint.
Use the other endpoints to manage conversations and retrieve information.

## Contributing
Contributions are welcome! Please create a pull request or open an issue for any improvements or bug fixes.

## License
This project is licensed under the MIT License.
