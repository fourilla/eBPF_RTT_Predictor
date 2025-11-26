import time
from flask import jsonify, render_template, Blueprint
from model.use_model import rtt_predictor

dashboard_bp = Blueprint('dashboard', __name__)

index = 0

@dashboard_bp.route("/")
def index_page():
    return render_template("dashboard.html")

@dashboard_bp.route("/predict")
def predict_once():
    from app import feature_service
    a = feature_service.get_latest()

    predictor = rtt_predictor(a)
    rtt_xgb = predictor.predict_data(a, model="xgboost")

    print(f"XGBoost 예측 RTT: {rtt_xgb:.2f} ms")

    return jsonify({
        "time": time.strftime("%H:%M:%S"),
        "actual": a["RTT_ms_mean"],
        "prediction": rtt_xgb,
    })
