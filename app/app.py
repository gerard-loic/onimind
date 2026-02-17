from flask import Flask, render_template, request, jsonify, Response

app = Flask("Onitama RL")

@app.route('/')
def index():
    return render_template('game.html')

if __name__ == '__main__':
    # Accessible depuis Windows sur http://localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
