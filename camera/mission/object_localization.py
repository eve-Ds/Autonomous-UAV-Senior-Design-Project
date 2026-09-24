#!/usr/bin/env python3
"""
Object Localization Mission.

Flies a lawnmower search pattern in offboard mode while running YOLO
detection on camera frames. When a target is detected, the drone's
GPS position is logged. Detections are deduplicated by proximity.

Two concurrent asyncio tasks handle flight and detection independently:
  - Flight task: NED velocity commands for the search pattern
  - Detection task: camera capture + YOLO inference + GPS tagging

Competition rules:
    - 3-4 large GCPs (~1.2m) define the search area
    - 5-10 smaller targets (~0.6m) with black numbers on white sections
    - Must not descend below 25ft (7.62m) AGL
    - 10 minutes to identify, classify, and localize all targets
    - Targets are survey-style ground control points (e.g. Amazon B07PHPFPJ3)

Config example (config/object_localization.json):
    {
        "altitude_m": 8.0,
        "min_altitude_m": 7.62,
        "search_speed_m_s": 1.5,
        "leg_spacing_m": 2.0,
        "timeout_s": 540,
        "search_area": [
            {"lat": 0.0, "lon": 0.0, "label": "GCP-A"},
            {"lat": 0.0, "lon": 0.0, "label": "GCP-B"},
            {"lat": 0.0, "lon": 0.0, "label": "GCP-C"},
            {"lat": 0.0, "lon": 0.0, "label": "GCP-D"}
        ],
        "yolo_model_path": "camera/yolo26n.pt",
        "yolo_confidence": 0.3,
        "yolo_classes": null,
        "detect_every_n_frames": 5
    }
"""

import asyncio
import logging
import math
import time
from pathlib import Path

from missions.base_mission import BaseMission
from camera.object_manager import ObjectManager



logger = logging.getLogger(__name__)

# Competition minimum altitude: 25 feet AGL (below this: -3 pts; below 20ft: flat 0)
MIN_ALTITUDE_FT = 25
MIN_ALTITUDE_M = MIN_ALTITUDE_FT * 0.3048   # 7.62m
ZERO_SCORE_ALTITUDE_FT = 20
ZERO_SCORE_ALTITUDE_M = ZERO_SCORE_ALTITUDE_FT * 0.3048  # 6.096m


class ObjectLocalizationMission(BaseMission):

    @property
    def name(self):
        return "object_localization"

    def _validate_config(self, config):
        alt = config.get("altitude_m", 8.0)
        min_alt = config.get("min_altitude_m", MIN_ALTITUDE_M)
        if alt < min_alt:
            raise ValueError(
                f"altitude_m ({alt}m) is below minimum allowed ({min_alt}m / {MIN_ALTITUDE_FT}ft AGL)"
            )
        # Must have either search_area (GPS corners) or grid dimensions
        has_area = "search_area" in config and len(config["search_area"]) >= 3
        has_grid = "grid_width_m" in config and "grid_length_m" in config
        if not has_area and not has_grid:
            raise ValueError(
                "Config must have either 'search_area' (3+ GPS coords) "
                "or 'grid_width_m'/'grid_length_m'"
            )

    @staticmethod
    def _gps_to_local(ref_lat, ref_lon, lat, lon):
        """Convert GPS to local NED offset (meters) relative to a reference point."""
        north = (lat - ref_lat) * 111139
        east = (lon - ref_lon) * 111139 * math.cos(math.radians(ref_lat))
        return north, east

    def _compute_search_bounds(self):
        """Compute search grid bounds from GPS search area corners.

        Returns (ref_lat, ref_lon, min_n, max_n, min_e, max_e) where
        N/E are local offsets in meters from the reference point.
        """
        area = self.config["search_area"]
        ref_lat = area[0]["lat"]
        ref_lon = area[0]["lon"]

        norths = []
        easts = []
        for pt in area:
            n, e = self._gps_to_local(ref_lat, ref_lon, pt["lat"], pt["lon"])
            norths.append(n)
            easts.append(e)

        # Add margin around the GCP-defined area
        margin = self.config.get("search_margin_m", 3.0)
        return (
            ref_lat, ref_lon,
            min(norths) - margin, max(norths) + margin,
            min(easts) - margin, max(easts) + margin,
        )

    def _generate_lawnmower_legs(self):
        """Generate a lawnmower (boustrophedon) search pattern.

        If search_area GPS coords are provided, computes the grid from them.
        Otherwise falls back to grid_width_m / grid_length_m.

        Returns a list of (north_vel, east_vel, duration_s) tuples.
        """
        speed = self.config["search_speed_m_s"]
        spacing = self.config.get("leg_spacing_m", 2.0)

        if "search_area" in self.config and len(self.config["search_area"]) >= 3:
            _, _, min_n, max_n, min_e, max_e = self._compute_search_bounds()
            length = max_n - min_n
            width = max_e - min_e
            logger.info(f"Search area from GCPs: {length:.1f}m x {width:.1f}m")
        else:
            length = self.config["grid_length_m"]
            width = self.config["grid_width_m"]
            logger.info(f"Search area from config: {length:.1f}m x {width:.1f}m")

        leg_duration = length / speed
        shift_duration = spacing / speed
        num_legs = max(1, int(width / spacing))

        legs = []
        direction = 1  # 1 = north, -1 = south

        for i in range(num_legs):
            legs.append((speed * direction, 0.0, leg_duration))
            direction *= -1
            if i < num_legs - 1:
                legs.append((0.0, speed, shift_duration))

        return legs

    async def _fly_to_search_start(self):
        """If using GPS search area, fly to the starting corner first."""
        if "search_area" not in self.config:
            return

        area = self.config["search_area"]
        ref_lat, ref_lon, min_n, max_n, min_e, max_e = self._compute_search_bounds()

        # Start corner is (min_n, min_e) — southwest-ish corner
        start_lat = ref_lat + min_n / 111139
        start_lon = ref_lon + min_e / (111139 * math.cos(math.radians(ref_lat)))

        logger.info(f"Flying to search start: ({start_lat:.6f}, {start_lon:.6f})")

        # Fly to start position using NED velocity (simple proportional)
        for _ in range(600):  # max 60 seconds at 10Hz
            lat, lon, alt = await self.controller.get_gps_position()
            n_err = (start_lat - lat) * 111139
            e_err = (start_lon - lon) * 111139 * math.cos(math.radians(lat))
            dist = math.sqrt(n_err ** 2 + e_err ** 2)

            if dist < 1.5:  # close enough
                logger.info("Reached search start position")
                await self.controller.hover()
                await asyncio.sleep(1)
                return

            speed = self.config["search_speed_m_s"]
            scale = min(speed / dist, speed)
            await self.controller.set_velocity_ned(
                north=n_err * scale, east=e_err * scale
            )
            await asyncio.sleep(0.1)

        logger.warning("Timed out flying to search start")

    async def _fly_search_pattern(self, stop_event):
        """Execute the lawnmower search pattern via NED velocity commands."""
        legs = self._generate_lawnmower_legs()
        logger.info(f"Search pattern: {len(legs)} legs")

        for i, (north_vel, east_vel, duration_s) in enumerate(legs):
            if stop_event.is_set() or not self.controller.is_safe_to_continue():
                logger.warning("Search pattern stopped early")
                return

            direction = "shift E" if east_vel > 0 else ("N" if north_vel > 0 else "S")
            logger.info(f"  Leg {i + 1}/{len(legs)}: {direction} for {duration_s:.1f}s")

            end_time = time.time() + duration_s
            while time.time() < end_time:
                if stop_event.is_set() or not self.controller.is_safe_to_continue():
                    return

                # Altitude safety check — do not descend below minimum
                _, _, alt = await self.controller.get_gps_position()
                min_alt = self.config.get("min_altitude_m", MIN_ALTITUDE_M)
                if alt < ZERO_SCORE_ALTITUDE_M:
                    logger.error(
                        f"Altitude {alt:.1f}m below {ZERO_SCORE_ALTITUDE_FT}ft "
                        f"— ZERO SCORE penalty threshold breached!"
                    )
                    self.result.data["zero_score_altitude_breach"] = True
                if alt < min_alt + 1: # +1 meter to add buffer before loosing points
                    logger.warning(
                        f"Altitude {alt:.1f}m approaching minimum {min_alt:.1f}m — climbing"
                    )
                    await self.controller.set_velocity_ned(
                        north=0.0, east=0.0, down=-0.5  # climb
                    )
                    await asyncio.sleep(0.5)
                    continue

                await self.controller.set_velocity_ned(north=north_vel, east=east_vel)
                await asyncio.sleep(0.1)

        await self.controller.hover()
        logger.info("Search pattern complete")


    async def _run_detection(self, stop_event):
        """Capture camera frames, run YOLO, log detections with GPS position.

        Runs concurrently with the flight task. Appends to the shared
        detections list when objects are found.
        """
        from ultralytics import YOLO
        from camera.tracking import mult_detect_targets   
        
                                 


        model_path = self.config.get("yolo_model_path", "camera/yolo26n.pt")
        confidence = self.config.get("yolo_confidence", 0.3)
        classes = self.config.get("yolo_classes", None)
        detect_every = self.config.get("detect_every_n_frames", 5)

        project_root = Path(__file__).resolve().parent.parent
        full_model_path = project_root / model_path

        model = YOLO(str(full_model_path))

        # Camera init — use Picamera2 on Pi, fall back to OpenCV webcam
        picam2 = None
        cap = None
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            cam_config = picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                controls={"FrameRate": 30}
            )
            picam2.configure(cam_config)
            picam2.start()
            await asyncio.sleep(2)
            logger.info("Camera: Picamera2 initialized")
        except ImportError:
            import cv2
            cap = cv2.VideoCapture(0)
            logger.info("Camera: OpenCV webcam fallback")

        frame_count = 0
        try:
            while not stop_event.is_set():
                if picam2:
                    frame = picam2.capture_array()
                elif cap:
                    ret, frame = cap.read()
                    if not ret:
                        await asyncio.sleep(0.05)
                        continue
                else:
                    break
                
                if frame_count % detect_every == 0:
                
                    targets, _ = mult_detect_targets(frame, model, classes, conf=confidence)

                    if targets:
                        lat, lon, alt = await self.controller.get_gps_position()
                        self.om.update(targets, (lat, lon), alt)
                
                frame_count += 1

                await asyncio.sleep(0)

        finally:
            if picam2:
                picam2.stop()
            if cap:
                cap.release()
            logger.info(f"Detection stopped after {frame_count} frames")

   

    async def pre_execute(self):
        """Arm, takeoff, then enter offboard mode for velocity control."""
        altitude = self.config.get("altitude_m", 8.0)

        self.om = ObjectManager() 

        # Enforce minimum altitude
        min_alt = self.config.get("min_altitude_m", MIN_ALTITUDE_M)
        if altitude < min_alt:
            logger.warning(
                f"Configured altitude {altitude}m below minimum {min_alt}m, "
                f"using {min_alt}m"
            )
            altitude = min_alt

        await self.controller.arm_and_takeoff(altitude)
        self.controller.start_monitors()
        await self.controller.start_offboard()

    async def execute(self):
        """Fly search pattern while running YOLO detection concurrently."""
        timeout = self.config.get("timeout_s", 540)  # 9 min default
        stop_event = asyncio.Event()


        # Fly to the start of the search area if GPS corners are defined
        await self._fly_to_search_start()

        # Launch detection as a background task
        detection_task = asyncio.create_task(
            self._run_detection(stop_event)
        )

        # Fly the search pattern with timeout
        try:
            await asyncio.wait_for(
                self._fly_search_pattern(stop_event),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Search timed out after {timeout}s")
            self.result.data["timed_out"] = True

        # Signal detection to stop and wait for cleanup
        stop_event.set()
        try:
            await asyncio.wait_for(detection_task, timeout=5.0)
        except asyncio.TimeoutError:
            detection_task.cancel()
            logger.warning("Detection task timed out during shutdown")

        # Deduplicate nearby detections
        unique = self.om.get_confirmed_objects()

        logger.info(f"\nUnique targets found:  {len(unique)}")
        for i, obj in enumerate(unique):
            logger.info(f"  Target {i + 1}: ({obj['lat']:.6f}, {obj['lon']:.6f}) class={obj['class']}")

        self.result.data["unique_targets"] = len(unique)
        self.result.data["targets"] = unique

    async def post_execute(self):
        """Stop offboard mode, then land."""
        try:
            await self.controller.drone.offboard.stop()
        except Exception:
            pass
        await self.controller.land()
        logger.info("-- Waiting for drone to land...")
        async for in_air in self.controller.drone.telemetry.in_air():
            if not in_air:
                logger.info("-- Drone is on the ground")
                break
        await asyncio.sleep(2)
        self.controller.stop_monitors()
