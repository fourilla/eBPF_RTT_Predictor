from flask import Flask

from feature_parser.feature_service import FeatureService
from pages.dashboard import dashboard_bp
from pages.monitor import monitor_bp
from pages.simulate import simulate_bp

app = Flask(__name__)

feature_service = FeatureService(nic_name="ens33")
feature_service.start() # feature 를 실시간으로 파싱하는 역할

app.register_blueprint(dashboard_bp)
app.register_blueprint(simulate_bp)
app.register_blueprint(monitor_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
