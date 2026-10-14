"""Flask/JWT/PostgreSQL reference for the feedback-learning exercise.

The exercise reads this file as text; it does not start the API.
To run the API separately, Flask, Flask-JWT-Extended, Flask-SQLAlchemy,
and a PostgreSQL driver such as psycopg2 are required. Set DATABASE_URL
to a postgresql+psycopg2:// connection URL and set JWT_SECRET_KEY to a
strong secret in the environment. Initialize tables before using the API.
"""

import os

from flask import Flask, jsonify, request
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt_identity,
    jwt_required,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash


db = SQLAlchemy()


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)


def create_app():
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ["DATABASE_URL"]
    app.config["JWT_SECRET_KEY"] = os.environ["JWT_SECRET_KEY"]
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    JWTManager(app)

    def credentials():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return None
        username = data.get("username")
        password = data.get("password")
        if not isinstance(username, str) or not username.strip():
            return None
        if not isinstance(password, str) or not password:
            return None
        return username.strip(), password

    @app.post("/register")
    def register():
        values = credentials()
        if values is None:
            return jsonify(error="Provide username and password as strings."), 400
        username, password = values
        if len(username) > 80 or len(password) < 8:
            return jsonify(error="Username max: 80 characters; password min: 8."), 400
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return jsonify(error="Username already exists."), 409
        return jsonify(id=user.id, username=user.username), 201

    @app.post("/login")
    def login():
        values = credentials()
        if values is None:
            return jsonify(error="Provide username and password as strings."), 400
        username, password = values
        user = db.session.execute(
            db.select(User).where(User.username == username)
        ).scalar_one_or_none()
        if user is None or not check_password_hash(user.password_hash, password):
            return jsonify(error="Invalid credentials."), 401
        return jsonify(access_token=create_access_token(identity=str(user.id)))

    @app.get("/profile")
    @jwt_required()
    def profile():
        try:
            user_id = int(get_jwt_identity())
        except (TypeError, ValueError):
            return jsonify(error="Invalid token identity."), 401
        user = db.session.get(User, user_id)
        if user is None:
            return jsonify(error="User not found."), 404
        return jsonify(id=user.id, username=user.username)

    @app.cli.command("init-db")
    def init_db():
        db.create_all()
        print("Database tables initialized.")

    return app


if __name__ == "__main__":
    create_app().run()
