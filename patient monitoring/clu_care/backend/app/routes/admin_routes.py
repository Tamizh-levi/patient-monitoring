# backend/app/routes/admin_routes.py
from flask import Blueprint, request, jsonify
from app.models.admin_model import Admin

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route("/users", methods=["GET"])
def get_all_users():
    users = Admin.get_all_users()
    return jsonify(users), 200


@admin_bp.route("/user/<role>/<user_id>", methods=["DELETE"])
def delete_user(role, user_id):
    success = Admin.delete_user(user_id, role)
    if success:
        return jsonify({"message": f"{role.capitalize()} deleted"}), 200
    else:
        return jsonify({"error": "User not found"}), 404


# Add this in your app.py or main file:
# from app.routes.admin_routes import admin_bp
# app.register_blueprint(admin_bp)
