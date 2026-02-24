from flask import Flask, render_template, request, jsonify, Response
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask("Onitama RL")

@app.route('/')
def index():
    return render_template('home.html', api_url=os.getenv('API_URL'), api_key=os.getenv('API_KEY'))

@app.route('/game')
def game():
    return render_template('game.html', api_url=os.getenv('API_URL'), api_key=os.getenv('API_KEY'))

if __name__ == '__main__':
    # Accessible depuis Windows sur http://localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
