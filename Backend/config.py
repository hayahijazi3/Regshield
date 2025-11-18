import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

db = SQLAlchemy()
jwt = JWTManager()

def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///app.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'dev-secret-key-change-me')

    CORS(
        app,
        resources={r"/*": {"origins": os.getenv('FRONTEND_ORIGIN', '*')}},
        supports_credentials=True,
        expose_headers=["Content-Type", "Authorization"],
    )

    db.init_app(app)
    jwt.init_app(app)
    return app
