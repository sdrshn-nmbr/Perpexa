from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
from dotenv import load_dotenv
import os
from openai import OpenAI
from exa_py import Exa

app = Flask(__name__)
# CORS(app, resources={r"/query": {"origins": "http://localhost:8000"}})
# CORS(app, resources={r"/query": {"origins": "http://localhost:3000"}})
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
exa_client = Exa(api_key=os.getenv("EXA_API_KEY"))

@app.route('/query', methods=['POST'])
def query():
    user_input = request.json['query']
    
    # Perform search using Exa AI
    search_results = exa_client.search_and_contents(
        user_input,
        type="neural",
        use_autoprompt=True,
        num_results=3,
        summary={
            "query": "Summarize the key points of this content."
        },
        category="general",
        exclude_domains=["en.wikipedia.org"],
        start_published_date="2023-01-01"
    )
    
    # Prepare context from search results
    context = "\n".join([f"Title: {result.title}\nContent: {result.text}\nSummary: {result.summary}" for result in search_results.results])
    
    # Generate AI response using OpenAI GPT-4
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's query."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuery: {user_input}"}
        ]
    )
    
    ai_response = response.choices[0].message.content

    # Prepare search results for frontend
    frontend_search_results = [
        {
            "title": result.title,
            "url": result.url,
            "summary": result.summary
        }
        for result in search_results.results
    ]

    return jsonify({
        "ai_response": ai_response,
        "search_results": frontend_search_results
    })

if __name__ == '__main__':
    app.run(debug=True)
