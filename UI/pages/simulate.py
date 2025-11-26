from flask import Blueprint

simulate_bp = Blueprint('simulate', __name__, url_prefix="/simulate")

@simulate_bp.route("/scenario1")
def test1():
    print("scenario1 실행됨")
    return "OK"

@simulate_bp.route("/scenario2")
def test2():
    print("scenario2 실행됨")
    return "OK"

@simulate_bp.route("/scenario3")
def test3():
    print("scenario3 실행됨")
    return "OK"
