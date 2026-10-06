"""Continuous QR decoding without ROS. One bounded worker per camera.

Camera callbacks never wait for OpenCV. Pending old frames are replaced by the
newest frame; capture-time pose and room generation travel with each result.
GUI and mission mutations belong to the ROS thread, not these workers.
"""
from dataclasses import dataclass
from queue import Queue, Empty, Full
from threading import Thread, Event
import time

import cv2
import numpy as np


@dataclass
class FrameJob:
    frame: np.ndarray
    captured_at: float
    position: tuple
    yaw: float
    front_range: float
    generation: int
    sequence: int


@dataclass
class DecodeResult:
    camera: str
    job: FrameJob
    decoded: list       # [(raw_text, corners in original image pixels)]
    outlines: list      # QR-looking contours, including undecoded ones
    finished_at: float
    elapsed: float
    error: str = ""


class RobustDecoder:
    """Multi and single QR, grayscale/contrast and bounded upscaling fallback."""

    def __init__(self):
        self.multi = cv2.QRCodeDetector()
        self.single = cv2.QRCodeDetector()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def decode(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variants = [(gray, 1.0)]
        # Keep a bounded amount of work so fresh frames are not delayed for seconds.
        enhanced = self.clahe.apply(gray)
        scale = min(2.0, 1600.0 / max(h, w))
        if scale > 1.05:
            variants.append((cv2.resize(enhanced, None, fx=scale, fy=scale,
                                        interpolation=cv2.INTER_CUBIC), scale))
        else:
            variants.append((enhanced, 1.0))
        decoded, outlines = {}, []
        for image, scale in variants:
            ok, texts, points, _ = self.multi.detectAndDecodeMulti(image)
            if points is not None:
                for text, corners in zip(texts, points):
                    corners = np.asarray(corners, np.float32).reshape(4, 2) / scale
                    outlines.append(corners)
                    if text:
                        decoded[text] = corners
            # Some OpenCV builds miss a single QR via the multi API.
            text, points, _ = self.single.detectAndDecode(image)
            if points is not None:
                corners = np.asarray(points, np.float32).reshape(4, 2) / scale
                outlines.append(corners)
                if text:
                    decoded[text] = corners
            if decoded:
                break
        return list(decoded.items()), outlines


class CameraWorker:
    def __init__(self, camera, output, decoder=None):
        self.camera = camera
        self.output = output
        self.pending = Queue(maxsize=1)
        self.stop_event = Event()
        self.decoder = decoder
        self.dropped = 0
        self.thread = Thread(target=self._run, name=f"qr-{camera}", daemon=True)
        self.thread.start()

    def submit(self, job):
        try:
            self.pending.put_nowait(job)
        except Full:
            try:
                self.pending.get_nowait()
                self.dropped += 1
            except Empty:
                pass
            try:
                self.pending.put_nowait(job)
            except Full:
                self.dropped += 1

    def _run(self):
        decoder = self.decoder or RobustDecoder()
        while not self.stop_event.is_set():
            try:
                job = self.pending.get(timeout=0.1)
            except Empty:
                continue
            start = time.monotonic()
            try:
                decoded, outlines = decoder.decode(job.frame)
                error = ""
            except Exception as exc:
                decoded, outlines, error = [], [], str(exc)
            result = DecodeResult(self.camera, job, decoded, outlines,
                                  time.monotonic(), time.monotonic() - start, error)
            try:
                self.output.put_nowait(result)
            except Full:
                try:
                    self.output.get_nowait()
                except Empty:
                    pass
                try:
                    self.output.put_nowait(result)
                except Full:
                    pass

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
