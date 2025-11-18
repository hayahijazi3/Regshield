from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from config import db
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(Integer, primary_key=True, autoincrement=True)
    username = db.Column(String(80), unique=True, nullable=False)
    email = db.Column(String(120), unique=True, nullable=False)
    password_hash = db.Column(String(256), nullable=False)
    phone_number = db.Column(String(20), nullable=True)
    linkedin_id = db.Column(String(255), nullable=True)

    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)


