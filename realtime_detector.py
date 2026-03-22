import argparse
import json
from pathlib import Path
from queue import Queue
from typing import Any

import joblib
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

from alerts import dispatch_alert
from storage import PredictionStorage

load_dotenv()


class RealtimeDetector:
    def __init__(
        self,
        model_path: str = "models/iot_model.joblib",
        metadata_path: str = "models/model_metadata.json",
        db_path: str = "data/predictions.db",
    ) -> None:
        self.model = joblib.load(model_path)
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self.feature_names: list[str] = metadata["feature_names"]
        self.attack_threshold: float = float(metadata.get("attack_threshold", 0.5))
        self.benign_labels: set[str] = set(
            val.lower() for val in metadata.get("benign_labels", ["benign", "normal", "0"])
        )
        self.events: Queue[dict[str, Any]] = Queue()
        self.storage = PredictionStorage(db_path=db_path)

    def _vectorize(self, record: dict[str, Any]) -> pd.DataFrame:
        row = {feature: record.get(feature, None) for feature in self.feature_names}
        return pd.DataFrame([row], columns=self.feature_names).replace([np.inf, -np.inf], 0)

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        vector = self._vectorize(record)
        pred = self.model.predict(vector)[0]
        probs = self.model.predict_proba(vector)[0] if hasattr(self.model, "predict_proba") else None

        attack_score = 0.0
        is_attack = str(pred).lower() not in self.benign_labels
        probabilities: dict[str, float] = {}
        if probs is not None and hasattr(self.model, "classes_"):
            probabilities = {str(c): float(p) for c, p in zip(self.model.classes_, probs)}
            benign_probs = [
                prob
                for klass, prob in probabilities.items()
                if str(klass).lower() in self.benign_labels
            ]
            benign_prob = max(benign_probs) if benign_probs else 0.0
            attack_score = 1.0 - benign_prob
            is_attack = attack_score >= self.attack_threshold

        result: dict[str, Any] = {
            "prediction": str(pred),
            "is_attack": is_attack,
            "attack_score": float(attack_score),
            "attack_threshold": self.attack_threshold,
        }
        if probabilities:
            result["probabilities"] = probabilities

        self.storage.save(record, result)
        if is_attack:
            dispatch_alert(
                f"IoT attack detected | prediction={result['prediction']} "
                f"| attack_score={result['attack_score']:.3f}"
            )
        self.events.put(result)
        return result

    def mqtt_consume(self, host: str, port: int, topic: str) -> None:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

        def on_connect(c: mqtt.Client, _userdata: Any, _flags: Any, reason_code: Any, _properties: Any) -> None:
            print(f"Connected to MQTT broker ({reason_code}), subscribing: {topic}")
            c.subscribe(topic)

        def on_message(_client: mqtt.Client, _userdata: Any, msg: mqtt.MQTTMessage) -> None:
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                result = self.predict(payload)
                print(f"[{topic}] {result}")
            except Exception as exc:
                print(f"Failed to process message: {exc}")

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(host, port, keepalive=60)
        client.loop_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MQTT real-time IoT detector.")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", default="iot/flow")
    parser.add_argument("--model-path", default="models/iot_model.joblib")
    parser.add_argument("--metadata-path", default="models/model_metadata.json")
    args = parser.parse_args()

    detector = RealtimeDetector(args.model_path, args.metadata_path)
    detector.mqtt_consume(args.mqtt_host, args.mqtt_port, args.mqtt_topic)


if __name__ == "__main__":
    main()
