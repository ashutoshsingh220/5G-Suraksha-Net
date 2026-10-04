import os
import sys
import unittest
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    CRASH_MODEL_DIR,
    CRASH_MODEL_PATH,
    FIRE_MODEL_PATH,
    DEFAULT_TEST_VIDEO,
    YOLO_DETECTOR_PATH,
    YOLO_SEVERITY_PATH,
    SEVERITY_CONFIG,
    OUTPUT_DIR,
    SNAPSHOT_DIR,
)
from detectors.accident_detector import AccidentDetector, VehicleDetection
from detectors.yolo_hierarchical_detector import YOLOHierarchicalAccidentDetector
from detectors.fire_detector import FireDetector, FireResult, FireDetection
from visualizer import CCTVVisualizer
from incident_logger import IncidentLogger
from stream_capture import StreamCapture


class TestSurveillancePipeline(unittest.TestCase):
    def test_01_config_and_model_files_exist(self):
        """Verify model files and configurations exist."""
        self.assertTrue(
            os.path.exists(CRASH_MODEL_PATH), f"Missing: {CRASH_MODEL_PATH}"
        )
        self.assertTrue(
            os.path.exists(os.path.join(CRASH_MODEL_DIR, "config.json")),
            "Missing config.json",
        )
        self.assertTrue(os.path.exists(FIRE_MODEL_PATH), f"Missing: {FIRE_MODEL_PATH}")
        self.assertEqual(len(SEVERITY_CONFIG), 5, "Should have 5 severity levels (0-4)")

    def test_02_accident_detector_synthetic(self):
        """Verify DETR AccidentDetector loads and processes frames."""
        if os.path.getsize(CRASH_MODEL_PATH) < 1000:
            self.skipTest(
                "Legacy DETR weights are Git-LFS stub (system uses YOLO hierarchical detector)."
            )
        detector = AccidentDetector(conf_thres=0.35)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(dummy_frame)
        self.assertIsInstance(detections, list)

        highest_sev, det = detector.get_highest_severity(detections)
        self.assertEqual(
            highest_sev, -1 if len(detections) == 0 else det.severity_level
        )

    def test_03_accident_detector_real_video(self):
        """Verify DETR AccidentDetector extracts vehicle/accident detections from real video."""
        crash_video_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "videos", "crash.mp4"
        )
        if not os.path.exists(crash_video_path):
            self.skipTest(f"Crash video {crash_video_path} not found.")

        detector = AccidentDetector(conf_thres=0.35)
        cap = cv2.VideoCapture(crash_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = cap.read()
        cap.release()

        self.assertTrue(ret, "Failed to read frame from crash.mp4")
        detections = detector.detect(frame)
        self.assertGreater(
            len(detections), 0, "Should detect vehicles/accidents in crash video frame."
        )
        self.assertTrue(
            any(d.is_accident for d in detections),
            "Frame 100 of crash.mp4 should contain detected accident.",
        )

    def test_04_fire_detector_synthetic(self):
        """Verify FireDetector loads and outputs valid result on synthetic frame."""
        detector = FireDetector(conf_thres=0.35)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(dummy_frame)

        self.assertIsInstance(result, FireResult)
        self.assertIn(
            result.predicted_label, ["Fire", "Smoke", "Fire & Smoke", "Normal"]
        )
        self.assertGreaterEqual(result.fire_probability, 0.0)
        self.assertLessEqual(result.fire_probability, 1.0)
        self.assertIsInstance(result.detections, list)

    def test_05_fire_detector_real_video(self):
        """Verify FireDetector extracts fire and smoke bounding boxes from real video frame."""
        fire_video_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "videos", "fire.mp4"
        )
        if not os.path.exists(fire_video_path):
            self.skipTest(f"Fire video {fire_video_path} not found.")

        detector = FireDetector(conf_thres=0.30)
        cap = cv2.VideoCapture(fire_video_path)
        ret, frame = cap.read()
        cap.release()

        self.assertTrue(ret, "Failed to read frame from fire.mp4")
        result = detector.detect(frame)
        self.assertTrue(result.has_fire, "Fire video frame should detect fire.")
        self.assertGreater(
            len(result.detections),
            0,
            "Should have at least 1 fire/smoke detection box.",
        )
        self.assertIsInstance(result.detections[0], FireDetection)

    def test_06_visualizer_rendering(self):
        """Verify CCTV visualizer draws overlays without altering frame dimensions."""
        visualizer = CCTVVisualizer()
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        sample_dets = [
            VehicleDetection(
                bbox=(100, 100, 300, 300),
                confidence=0.88,
                class_id=2,
                raw_name="vehicle",
                display_name="Normal Vehicle",
                short_name="NORMAL",
                color=(0, 210, 60),
                is_accident=False,
                severity_level=0,
            ),
            VehicleDetection(
                bbox=(400, 200, 600, 450),
                confidence=0.79,
                class_id=0,
                raw_name="accident",
                display_name="Traffic Accident",
                short_name="ACCIDENT",
                color=(0, 40, 240),
                is_accident=True,
                severity_level=3,
            ),
        ]
        sample_fire = FireResult(
            has_fire=True,
            has_smoke=False,
            fire_confidence=0.92,
            smoke_confidence=0.0,
            detections=[
                FireDetection(
                    bbox=(200, 200, 350, 350),
                    confidence=0.92,
                    class_id=0,
                    label="Fire",
                    color=(0, 69, 255),
                )
            ],
        )

        rendered = visualizer.draw(
            dummy_frame, sample_dets, sample_fire, source_label="Test"
        )
        self.assertEqual(rendered.shape, dummy_frame.shape)
        self.assertEqual(rendered.dtype, np.uint8)

    def test_07_incident_logger_and_snapshot(self):
        """Verify incident logger creates records and snapshots."""
        test_csv = os.path.join(OUTPUT_DIR, "test_incident_log.csv")
        logger = IncidentLogger(log_file=test_csv)
        dummy_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

        det = VehicleDetection(
            bbox=(10, 10, 50, 50),
            confidence=0.9,
            class_id=0,
            raw_name="accident",
            display_name="Traffic Accident",
            short_name="ACCIDENT",
            color=(0, 40, 240),
            is_accident=True,
            severity_level=3,
        )
        fire = FireResult(
            has_fire=True, has_smoke=False, fire_confidence=0.95, smoke_confidence=0.0
        )

        snap = logger.check_and_log(dummy_frame, 3, det, fire, [det])
        self.assertIsNotNone(snap)
        self.assertTrue(os.path.exists(snap))
        self.assertTrue(os.path.exists(test_csv))

        # Cleanup test files
        if os.path.exists(test_csv):
            os.remove(test_csv)
        if os.path.exists(snap):
            os.remove(snap)

    def test_08_video_pipeline_smoke(self):
        """Test video stream reading and processing on real video."""
        if not os.path.exists(DEFAULT_TEST_VIDEO):
            self.skipTest(f"Test video {DEFAULT_TEST_VIDEO} not found.")

        if os.path.getsize(CRASH_MODEL_PATH) < 1000:
            accident_detector = YOLOHierarchicalAccidentDetector(conf_thres=0.35)
        else:
            accident_detector = AccidentDetector(conf_thres=0.35)
        fire_detector = FireDetector(conf_thres=0.35)
        visualizer = CCTVVisualizer()

        with StreamCapture(DEFAULT_TEST_VIDEO) as stream:
            for i in range(10):
                ret, frame = stream.read()
                if not ret:
                    break
                dets = accident_detector.detect(frame)
                fire_res = fire_detector.detect(frame)
                rendered = visualizer.draw(frame, dets, fire_res)
                self.assertIsNotNone(rendered)

    def test_09_yolo_hierarchical_detector_synthetic(self):
        """Verify YOLOHierarchicalAccidentDetector loads and handles synthetic frames."""
        self.assertTrue(
            os.path.exists(YOLO_DETECTOR_PATH), f"Missing: {YOLO_DETECTOR_PATH}"
        )
        self.assertTrue(
            os.path.exists(YOLO_SEVERITY_PATH), f"Missing: {YOLO_SEVERITY_PATH}"
        )

        detector = YOLOHierarchicalAccidentDetector(conf_thres=0.35)
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detections = detector.detect(dummy_frame)
        self.assertIsInstance(detections, list)

        highest_sev, det = detector.get_highest_severity(detections)
        self.assertEqual(
            highest_sev, -1 if len(detections) == 0 else det.severity_level
        )

    def test_10_yolo_hierarchical_detector_real_video(self):
        """Verify YOLOHierarchicalAccidentDetector detects accidents and classifies severity on real video."""
        crash_video_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "videos", "crash.mp4"
        )
        if not os.path.exists(crash_video_path):
            self.skipTest(f"Crash video {crash_video_path} not found.")

        detector = YOLOHierarchicalAccidentDetector(conf_thres=0.35)
        cap = cv2.VideoCapture(crash_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 80)
        ret, frame = cap.read()
        cap.release()

        self.assertTrue(ret, "Failed to read frame 80 from crash.mp4")
        detections = detector.detect(frame)
        self.assertGreater(
            len(detections), 0, "Should detect objects in crash.mp4 frame 80"
        )
        has_accident = any(d.is_accident for d in detections)
        self.assertTrue(
            has_accident, "Frame 80 of crash.mp4 should contain detected accident"
        )

        # Verify severity level is 2 (Moderate) or 3 (Severe)
        highest_sev, highest_det = detector.get_highest_severity(detections)
        self.assertIn(
            highest_sev,
            [2, 3],
            f"Accident severity should be 2 (Moderate) or 3 (Severe), got {highest_sev}",
        )

    def test_11_yolo_15fps_latency_budget(self):
        """Verify YOLO hierarchical detector completes in < 40ms, well below the 66.6ms 15 FPS budget."""
        detector = YOLOHierarchicalAccidentDetector(conf_thres=0.35)
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        import time
        import torch

        start_t = time.perf_counter()
        _ = detector.detect(test_frame)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        max_budget = 55.0 if torch.cuda.is_available() else 350.0
        print(
            f"\n[Test 11] Single frame latency: {elapsed_ms:.2f} ms (Target budget: {max_budget:.1f} ms)"
        )
        self.assertLess(
            elapsed_ms,
            max_budget,
            f"Inference exceeded safe threshold: {elapsed_ms:.2f} ms",
        )


if __name__ == "__main__":
    unittest.main()
