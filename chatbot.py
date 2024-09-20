import base64
import os
import requests
from flask import Flask, request, jsonify, session
from flask_pymongo import MongoClient
from bson.objectid import ObjectId
import datetime
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util


#Environment Work
hugging_face_key=os.environ.get('hugging_face_key')
connection_string = os.environ.get('MONGODB_URI')
app = Flask(__name__)
app.secret_key = 'hackerx123'

#API Routes
@app.route('/',methods=['GET'])
def ind():
   return 'System is ON and Working at Port 5000 at http://127.0.0.1:5000/api/ENDPOINTS with api endoint chat, test'


@app.route('/test',methods=['GET'])
def test():
   return 'System is ON and Working at Port 5000 at http://127.0.0.1:5000/api/ENDPOINTS with api endoint chat, test'

@app.route('/chat',methods=['GET','POST'])
def chat():
    """
    Handles chat interactions via POST requests.
    User input should be sent as JSON: { "user_input": "<message>" }
    """
    
    # Get user input from the POST request
    user_input = request.json.get('user_input', '').strip()
    
    if not user_input:
        return jsonify({"error": "No user input provided"}), 400
    
    # Handle user input to generate a response
    function_response = handle_function_calls(user_input)
    
    if function_response:
        response = function_response
        add_to_history(user_input, function_response)
    else:
        fallback = fallback_response(user_input)
        response = fallback
        add_to_history(user_input, fallback)
    
    # Return the bot's response as JSON
    return jsonify({"response": response})

#Databse

# MongoDB Configuration
# Replace with your actual MongoDB Atlas connection string

client = MongoClient(connection_string)
db = client["chatbot"]

# MongoDB Collections
conversations_collection = db["conversations"]
users_collection = db["users_conversations"]

def get_user_id():
    if 'user_id' not in session:
        # Check if the session already has a user ID; if not, create a new user entry
        existing_user = users_collection.find_one({'session_id': session.get('session_id')})
        if existing_user:
            session['user_id'] = str(existing_user['_id'])
        else:
            # Create a new user entry if not found
            user_id = ObjectId()
            session['user_id'] = str(user_id)
            users_collection.insert_one({'_id': user_id, 'session_id': session.get('session_id'),'email': 'priyanshu.k4444@gmail.com'})
    return session['user_id']

def serialize_conversation(conversation):
    # Convert ObjectId to string
    conversation['user_id'] = str(conversation['user_id'])
    return conversation

@app.route('/conversations', methods=['POST'])
def create_conversation():
    data = request.get_json()
    if not data or 'prompt' not in data or 'answer' not in data:
        return jsonify({'error': 'Missing required fields (prompt and answer)'}), 400

    user_id = get_user_id()
    conversation = {
        'user_id': user_id,
        'prompt': data['prompt'],
        'answer': data['answer'],
        'timestamp': datetime.datetime.utcnow(),
        'email': 'priyanshu.k4444@gmail.com'
    }

    result = conversations_collection.insert_one(conversation)
    result = users_collection.insert_one(conversation)
    return jsonify({'conversation_id': str(result.inserted_id)}), 201

@app.route('/conversations/getsession', methods=['GET'])
def get_conversations():
    user_id = str(get_user_id())
    conversations = list(conversations_collection.find({'user_id': user_id}))
    #serialized_conversations = [serialize_conversation(convo) for convo in conversations]

    # Serialize each conversation to convert ObjectId to string
    serialized_conversations = []
    for convo in conversations:
        convo['_id'] = str(convo['_id'])  # Convert MongoDB ObjectId to string
        convo['user_id'] = str(convo['user_id'])  # Convert user_id to string
        serialized_conversations.append(convo)

    return jsonify(serialized_conversations)

    
    
@app.route('/conversations/get/<user_id>', methods=['GET'])
def get_conversations_by_user_id(user_id):
    # Ensure the user_id is a string, not an ObjectId
    user_id = str(user_id)
    
    # Query the database to get conversations for the given user_id
    conversations = list(conversations_collection.find({'user_id': user_id}))

    # Serialize each conversation to convert ObjectId to string
    serialized_conversations = []
    for convo in conversations:
        convo['_id'] = str(convo['_id'])  # Convert MongoDB ObjectId to string
        convo['user_id'] = str(convo['user_id'])  # Convert user_id to string
        serialized_conversations.append(convo)

    return jsonify(serialized_conversations)


#this require object id
@app.route('/conversations/update/<conversation_id>', methods=['PUT'])
def update_conversation(conversation_id):
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing data'}), 400

    try:
        conversation_id = ObjectId(conversation_id)
    except Exception:
        return jsonify({'error': 'Invalid conversation ID'}), 400

    # Add the '_id' field to the update data
    data['_id'] = conversation_id

    # Update the document using the '_id' field
    result = conversations_collection.update_one(
        {'_id': conversation_id},
        {'$set': data}
    )

    if result.modified_count == 1:
        return jsonify({'message': 'Conversation updated successfully'})
    else:
        return jsonify({'error': 'Conversation not found or not updated'}), 404


@app.route('/conversations/delete/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    user_id = get_user_id()
    try:
        conversation_id = ObjectId(conversation_id)
    except Exception:
        return jsonify({'error': 'Invalid conversation ID'}), 400

    result = conversations_collection.delete_one(
        {'_id': conversation_id, 'user_id': user_id}
    )
    if result.deleted_count == 1:
        return jsonify({'message': 'Conversation deleted successfully'})
    else:
        return jsonify({'error': 'Conversation not found or not deleted'}), 404

@app.route('/knowledge_base', methods=['GET'])
def get_knowledge_base():
    conversations = list(conversations_collection.find())
    serialized_conversations = []

    for convo in conversations:
        # Handle potential absence of 'user_id' key
        user_id = convo.get('user_id')  # Use get() to avoid KeyError
        if user_id:
            convo['user_id'] = str(user_id)  # Convert user_id to string only if it exists

        # Convert MongoDB ObjectId to string for '_id' field
        convo['_id'] = str(convo['_id'])

        serialized_conversations.append(convo)

    return jsonify(serialized_conversations)


@app.route('/session', methods=['GET'])
def get_session_info():
    user_id = get_user_id()
    return jsonify({'user_id': user_id})

@app.route('/logout', methods=['POST'])
def logout():
    user_id = session.get('user_id')
    if user_id:
        # Delete all objects associated with the user_id
        conversations_collection.delete_many({'user_id': user_id})
        session.pop('user_id', None)
        session.pop('session_id', None)

    return jsonify({'message': 'Logged out successfully'})

#Functions 

API_URL = "https://api-inference.huggingface.co/models/impira/layoutlm-invoices"
headers = {"Authorization": f"Bearer {hugging_face_key}"}

def query(payload):
		
	with open(payload["image"], "rb") as f:
		img = f.read()
		payload["image"] = base64.b64encode(img).decode("utf-8")
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

output = query({
	"image": "uploads/invv.png",
	"question": "What is the tax amount?"
})

##SANDDARBH CODE
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Step 2: Initialize Models and Define Intents
intent_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
intent_classifier = pipeline("text-classification", model=intent_model_name)

similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

INTENTS = {
    "get_time": {
        "examples": ["What time is it?", "Tell me the current time", "Can you give me the time?","what is the time clock showing ?"],
        "action": "get_current_time"
    },
    "add_numbers": {
        "examples": ["Add two and three", "What is the sum of 10 and 5?", "Calculate 7 plus 8"],
        "action": "perform_addition"
    },
    "weather_forecast": {
        "examples": ["What's the weather like?", "Tell me the forecast", "Is it going to rain today?"],
        "action": "weather_forecast"
    },
    "data_retrieval": {
    "examples": [
        "Can you fetch my order details?",
        "Retrieve my invoices for this month",
        "Show me the pending payments",
        "Get my order history",
        "Fetch the details of my last purchase",
        "What are the details of invoice #1234?",
        "Retrieve the current status of my order",
        "Can you show me my delivery address?",
        "Get my chat details"
        "Get my account details"
    ],
    "action": "data_retrieval"
    },
    "data_save_operations": {
    "examples": [
        "Create a new order for me",
        "Can you save my new delivery address?",
        "Add a new item to my cart",
        "Update the invoice status to paid",
        "Save my contact information",
        "Modify the quantity of my order",
        "Update my account preferences",
        "Change my subscription plan",
        "Record this payment in the system",
        "Save this transaction to my account"
    ],
    "action": "data_save_operations"
}


}


conversation_history = []

# Step 3: Define Action Functions
def data_retrieval(user_id):
    url = f"http://127.0.0.1:5000/conversations/get/{user_id}"  # Replace with your actual API base URL
    response = requests.get(url)

    if response.status_code == 200:
        out=str(response.json())# Returns the conversations as a list of dictionaries (JSON)
        return out  
    else:
        return f"Error: {response.status_code}"
    



def get_current_time():
    """Returns the current time."""
    now = datetime.datetime.now()
    return f"The current time is {now.strftime('%H:%M:%S')}."

def perform_addition(a, b):
    """Performs addition of two numbers."""
    return f"The result of adding {a} and {b} is {a + b}."

def weather_forecast(location="New York"):
    """Fetches real weather data using OpenWeatherMap API."""
    API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with your actual API key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    
    complete_url = f"{base_url}appid={API_KEY}&q={location}&units=metric"
    response = requests.get(complete_url)
    
    if response.status_code == 200:
        data = response.json()
        weather = data['weather'][0]['description']
        temperature = data['main']['temp']
        return f"The current weather in {location} is {weather} with a temperature of {temperature}°C."
    else:
        return "I'm sorry, I couldn't fetch the weather information right now."

# Step 4: Intent Recognition
def recognize_intent(user_input):
    """Recognizes the intent of the user input."""
    max_similarity = 0
    recognized_intent = None
    
    for intent, details in INTENTS.items():
        for example in details["examples"]:
            similarity = util.pytorch_cos_sim(
                similarity_model.encode(user_input, convert_to_tensor=True),
                similarity_model.encode(example, convert_to_tensor=True)
            ).item()
            if similarity > max_similarity:
                max_similarity = similarity
                recognized_intent = intent
    
    threshold = 0.6
    if max_similarity >= threshold:
        return recognized_intent
    else:
        return "unknown"

# Step 5: Entity Extraction
def extract_numbers(user_input):
    """Extracts numbers from the user input."""
    return [int(word) for word in user_input.split() if word.isdigit()]

def extract_location(user_input):
    """Extracts location from the user input."""
    # Placeholder implementation
    # Enhance this function to extract actual locations using NLP techniques or APIs
    return "New York"

# Step 6: Handle Function Calls
def handle_function_calls(user_input):
    intent = recognize_intent(user_input)
    
    if intent == "get_time":
        return get_current_time()
    
    elif intent == "add_numbers":
        numbers = extract_numbers(user_input)
        if len(numbers) >= 2:
            return perform_addition(numbers[0], numbers[1])
        else:
            return "Please provide at least two numbers to add."
    
    elif intent == "weather_forecast":
        location = extract_location(user_input)
        return weather_forecast(location)
    elif intent == "data_retrieval":
        user_id = "66ed27a0012ef88acc5b1cd4"
        return data_retrieval(user_id)
    
    else:
        return None

# Step 7: Contextual Awareness
def add_to_history(user_input, bot_response):
    """Adds the user input and bot response to the conversation history."""
    conversation_history.append({"user": user_input, "bot": bot_response})

def get_context():
    """Retrieves the last few interactions for context."""
    return conversation_history[-3:]

# Step 8: Fallback with AI
# fallback_model_name = "EleutherAI/gpt-neo-1.3B" #can change it later
fallback_model_name = "microsoft/phi-1.5" #can change it later
fallback_generator = pipeline("text-generation", model=fallback_model_name, max_length=50)

def fallback_response(user_input):
    """Generates a fallback response using a generative model."""
    # prompt = f"The user said: '{user_input}'. The bot does not understand and needs to respond helpfully."
    prompt = f"Provide a helpful response to this message: '{user_input}'."

    response = fallback_generator(prompt, max_length=50, num_return_sequences=1)
    return response[0]['generated_text'].split("The bot")[0].strip()

# # Step 9: Main Interaction Function
# def interact_with_model():
#     print("Chatbot ready. Type 'exit' to end the conversation.")
#     while True:
#         user_input = input("User: ").strip()
    
#         if user_input.lower() == 'exit':
#             print("Bot: Goodbye!")
#             break
    
#         function_response = handle_function_calls(user_input)
        
#         if function_response:
#             print(f"Bot: {function_response}")
#             add_to_history(user_input, function_response)
#         else:
#             fallback = fallback_response(user_input)
#             print(f"Bot: {fallback}")
#             add_to_history(user_input, fallback)



# print()


# interact_with_model()