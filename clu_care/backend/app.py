import sys
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))
from app.models.admin_model import Admin

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    role = data.get('role')
    
    # You would lookup user in DB based on role/email, check password
    # Here is a simple dummy check for example
    if role == 'admin' and email == 'admin@example.com' and password == '123@admin':
        return jsonify({"message": "Login successful", "role": role}), 200
    else:
        return jsonify({"message": "Invalid credentials"}), 401

# Example test route to check backend is running
@app.route('/')
def index():
    return jsonify({"message": "Flask backend is running"}), 200

# Example admin users route
@app.route('/admin/users')
def get_admin_users():
    admins = Admin.get_all()  # Make sure your Admin model has get_all method
    # Serialize admins to list of dicts
    admins_list = [admin.to_dict() for admin in admins]
    return jsonify(admins_list), 200

# Run server
if __name__ == '__main__':
    app.run(debug=True)
