from feature_parser.feature_collector import FinalRawCollector, FeatureTransformer
import threading, time

class FeatureService:
    def __init__(self, nic_name="ens33"):
        self.collector = FinalRawCollector()
        self.transformer = FeatureTransformer(nic_name=nic_name)
        self.latest_feature = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._update_loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def _update_loop(self):
        while not self.stop_event.is_set():
            raw = self.collector.get_snapshot()
            feat = self.transformer.transform(raw)
            with self.lock:
                self.latest_feature = feat
            time.sleep(1)

    def get_latest(self):
        with self.lock:
            return self.latest_feature.copy()
