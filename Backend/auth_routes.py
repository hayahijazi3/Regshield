from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from models import User
from config import db

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json(force=True)
    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not username or not email or not password:
        return jsonify({"msg": "username, email, password required"}), 400

    if User.query.filter((User.email==email)|(User.username==username)).first():
        return jsonify({"msg": "user with that email/username exists"}), 409

    u = User(username=username, email=email)
    u.set_password(password)
    db.session.add(u)
    db.session.commit()
    return jsonify({"msg": "created"}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json(force=True)
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not email or not password:
        return jsonify({"msg": "Email and password are required."}), 400
    user = User.query.filter_by(email=email).first()
    if user is None or not user.check_password(password):
        return jsonify({"msg": "Invalid email or password."}), 401
    token = create_access_token(identity=str(user.id), additional_claims={"username": user.username, "email": user.email})
    return jsonify({"access_token": token}), 200

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    user_id = get_jwt_identity()
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return jsonify({"msg": "Invalid token subject."}), 422
    user = User.query.get(user_id)
    if not user:
        return jsonify({"msg": "User not found."}), 404
    return jsonify({"id": user.id, "username": user.username, "email": user.email, "phone_number": user.phone_number}), 200
