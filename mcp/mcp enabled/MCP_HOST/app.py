from flask import Flask, render_template, request, jsonify
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the personal fitness assistant functionality
from personal_fitness_assistant.main import query_fitness_assistant, initialize_fitness_assistant
from personal_fitness_assistant.utils import save_user_query

# Initialize Flask app
app = Flask(__name__)

# Flag to track if this is the first request
_fitness_assistant_initialized = False

# Initialize the fitness assistant on first request
@app.before_request
def initialize_on_first_request():
    global _fitness_assistant_initialized
    if not _fitness_assistant_initialized:
        initialize_fitness_assistant()
        _fitness_assistant_initialized = True

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# API route for fitness queries
@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({'error': 'Query is required'}), 400
    
    # Get response from fitness assistant
    response = query_fitness_assistant(query_text)
    
    # Save the query and response (using a default user ID for now)
    save_user_query('default_user', query_text, response)
    
    return jsonify({
        'query': query_text,
        'response': response
    })

# Run the app if executed directly
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)