# backend/app/models/admin_model.py
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash
from app.utils.db import db  # your MongoDB client instance


class Admin:
    collection = db.admins

    @staticmethod
    def create_admin(username, email, password):
        if Admin.collection.find_one({"email": email}):
            return None  # Admin already exists
        hashed_pw = generate_password_hash(password)
        admin = {
            "username": username,
            "email": email,
            "password": hashed_pw,
            "role": "admin"
        }
        result = Admin.collection.insert_one(admin)
        admin["_id"] = str(result.inserted_id)
        return admin

    @staticmethod
    def find_by_email(email):
        admin = Admin.collection.find_one({"email": email})
        if admin:
            admin["_id"] = str(admin["_id"])
        return admin

    @staticmethod
    def verify_password(admin, password):
        return check_password_hash(admin["password"], password)

    @staticmethod
    def get_all_users():
        users = []
        for role in ["patients", "doctors", "pharmacies"]:
            coll = db[role]
            for user in coll.find():
                user["_id"] = str(user["_id"])
                user["role"] = role[:-1]  # singular role
                users.append(user)
        return users

    @staticmethod
    def delete_user(user_id, role):
        coll = db[role + "s"]
        result = coll.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count > 0
