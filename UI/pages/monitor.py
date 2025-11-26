from flask import Blueprint, jsonify, render_template

monitor_bp = Blueprint('monitor', __name__, url_prefix="/monitor")

@monitor_bp.route('/')
def net():
    return render_template('monitor.html')

@monitor_bp.route('/api')
def get_net():
    from app import feature_service
    return jsonify(feature_service.get_latest())