from flask import Flask, render_template, jsonify
from ml_service import run_comparison

app = Flask(__name__)

@app.route('/')
def home():
    # Frontend HTML page load karega
    return render_template('index.html')

@app.route('/compare', methods=['GET'])
def compare():
    try:
        results = run_comparison()
        return jsonify({
            "status": "success",
            "comparison_results": results
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)