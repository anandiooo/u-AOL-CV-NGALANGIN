from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from features import detect_fast_keypoints, verify_gerobak_mask
from geometry import (
	compute_accessibility,
	compute_area_m2,
	compute_homography,
	pixel_to_meter_ratio,
	warp_mask,
)
from perception import DualModelPerception
from processing import clean_mask

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


st.set_page_config(
	page_title="Road Capacity & Sidewalk Dashboard",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {
	font-family: 'Inter', sans-serif;
	font-size: 16px;
}
:root {
	--primary:   #615fff;
	--surface:   #1d293d;
	--surface-2: #0f172b;
	--border:    #314158;
	--text-main: #e2e8f0;
	--text-muted:#94a3b8;
	--radius:    8px;
	--green:     #10b981;
	--yellow:    #f59e0b;
	--orange:    #f97316;
	--red:       #ef4444;
}
:root, [data-theme="light"], [data-theme="dark"] {
	--primary-color: var(--primary) !important;
	--background-color: var(--surface-2) !important;
	--secondary-background-color: var(--surface) !important;
	--text-color: var(--text-main) !important;
}
body { background-color: var(--surface-2) !important; color: var(--text-main) !important; }
[data-testid="stAppViewContainer"] { background-color: var(--surface-2) !important; }
[data-testid="stHeader"] { height: 0px !important; background: transparent !important; }
.main .block-container {
	padding: 1rem 2rem !important;
	max-width: 1400px;
}
.metric-card {
	background: var(--surface);
	border: 1px solid var(--border);
	border-radius: var(--radius);
	padding: 1rem 1.2rem;
	transition: all 0.2s ease-in-out;
}
.metric-card:hover {
	transform: translateY(-2px);
	border-color: var(--primary);
	box-shadow: 0 6px 16px rgba(0,0,0,0.2);
}
.metric-card .label {
	font-size: .8rem; font-weight: 500;
	text-transform: uppercase; letter-spacing: .08em;
	color: var(--text-muted); margin-bottom: .25rem;
}
.metric-card .value {
	font-size: 1.7rem; font-weight: 600;
	color: var(--text-main); letter-spacing: -0.5px;
	font-family: 'JetBrains Mono', monospace;
}
.metric-card .sub {
	font-size: .78rem; color: var(--text-muted); margin-top: .15rem;
}
.section-heading {
	font-size: 1rem; font-weight: 500;
	color: var(--text-muted); text-transform: uppercase;
	letter-spacing: .06em; margin: 1.2rem 0 .6rem;
}
.stTextInput > div > div > input,
.stSelectbox [data-baseweb="select"],
.stNumberInput > div > div > input {
	border-radius: 8px !important;
	background-color: var(--surface) !important;
	border: 1px solid var(--border) !important;
	color: var(--text-main) !important;
}
div.stButton > button[kind="primary"] {
	background: var(--primary) !important;
	border: none !important; border-radius: 8px !important;
	font-weight: 500 !important;
	color: white !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[kind="primary"]:focus,
div.stButton > button[kind="primary"]:active {
	background: #4f46e5 !important;
	color: white !important;
	border: none !important;
	box-shadow: none !important;
}
div.stButton > button:not([kind="primary"]) {
	border-radius: 8px !important;
	background-color: var(--surface) !important;
	border: 1px solid var(--border) !important;
	color: var(--text-main) !important;
}
div.stButton > button:not([kind="primary"]):hover,
div.stButton > button:not([kind="primary"]):focus,
div.stButton > button:not([kind="primary"]):active {
	border-color: var(--primary) !important;
	color: var(--primary) !important;
	background-color: var(--surface) !important;
	box-shadow: none !important;
}
/* Sidebar layout consistency styling */
[data-testid="stSidebar"] {
	background-color: var(--surface-2) !important;
	border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
	background-color: var(--surface) !important;
	border: 1px solid var(--border) !important;
	border-radius: var(--radius) !important;
	margin-bottom: 0.8rem !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
	color: var(--text-main) !important;
	font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
	background-color: var(--surface) !important;
	color: var(--text-main) !important;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
	background-color: var(--surface-2) !important;
	border: 1px solid var(--border) !important;
	color: var(--text-main) !important;
	border-radius: 8px !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .section-heading {
	color: var(--text-muted) !important;
	font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
	background-color: var(--surface-2) !important;
	border: 1px dashed var(--border) !important;
	border-radius: 8px !important;
	padding: 10px !important;
}
/* Active tab underline and text color (no Streamlit default red) */
[data-baseweb="tab-list"] button[aria-selected="true"] {
	color: var(--primary) !important;
	border-bottom-color: var(--primary) !important;
}
[data-baseweb="tab-highlight"] {
	background-color: var(--primary) !important;
}
button[data-baseweb="tab"] div p {
	color: var(--text-muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"] div p {
	color: var(--primary) !important;
}
/* Override links to prevent default red */
a {
	color: var(--primary) !important;
}
a:hover {
	color: #4f46e5 !important;
}
</style>
""",
	unsafe_allow_html=True,
)
if "running" not in st.session_state:
	st.session_state["running"] = False
if "paused" not in st.session_state:
	st.session_state["paused"] = False
if "video_frame_index" not in st.session_state:
	st.session_state["video_frame_index"] = 0
if "model" not in st.session_state:
	st.session_state["model"] = None
if "model_key" not in st.session_state:
	st.session_state["model_key"] = None
if "frame_width" not in st.session_state:
	st.session_state["frame_width"] = 640
if "frame_height" not in st.session_state:
	st.session_state["frame_height"] = 480

# metric card
def metric_card(label: str, value: str, sub: str = "") -> str:
	sub_html = f'<div class="sub">{sub}</div>' if sub else ""
	return (
		f'<div class="metric-card">'
		f'<div class="label">{label}</div>'
		f'<div class="value">{value}</div>'
		f'{sub_html}'
		f'</div>'
	)

# click start resume
def click_start_resume():
	st.session_state["running"] = True
	st.session_state["paused"] = False

# click pause
def click_pause():
	st.session_state["paused"] = True

# click stop
def click_stop():
	st.session_state["running"] = False
	st.session_state["paused"] = False
	st.session_state["video_frame_index"] = 0
	if "last_results" in st.session_state:
		del st.session_state["last_results"]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL_A = MODEL_DIR / "yolov11_obstacle_model" / "weights" / "best.pt"
DEFAULT_MODEL_B = MODEL_DIR / "yolov11x-seg.pt"

# get accessibility color
def get_accessibility_color(accessibility: float) -> str:
	if accessibility >= 80.0:
		return "var(--green)"
	elif accessibility >= 50.0:
		return "var(--yellow)"
	return "var(--orange)"

# get accessibility rating
def get_accessibility_rating(accessibility: float) -> str:
	if accessibility >= 80.0:
		return "Accessible"
	elif accessibility >= 50.0:
		return "Partially Blocked"
	return "Critically Blocked"

# get available models
def get_available_models() -> dict[str, Path]:
	pt_files = list(MODEL_DIR.glob("**/*.pt"))
	models_dict = {}
	for path in pt_files:
		try:
			rel = path.relative_to(MODEL_DIR)
			name = str(rel).replace("\\", "/")
		except ValueError:
			name = path.name
		models_dict[name] = path
	default_a_name = "yolov11_obstacle_model/weights/best.pt"
	if default_a_name not in models_dict and DEFAULT_MODEL_A.exists():
		models_dict[default_a_name] = DEFAULT_MODEL_A
	default_b_name = "yolov11x-seg.pt"
	if default_b_name not in models_dict and DEFAULT_MODEL_B.exists():
		models_dict[default_b_name] = DEFAULT_MODEL_B
	return models_dict

@st.cache_resource
# get ensemble classes
def get_ensemble_classes(model_a_path_str: str, model_b_path_str: str) -> list[str]:
	classes = set()
	try:
		from ultralytics import YOLO
		try:
			m_a = YOLO(model_a_path_str)
			classes.update(m_a.names.values())
		except Exception:
			classes.add("gerobak")
		try:
			m_b = YOLO(model_b_path_str)
			classes.update(m_b.names.values())
		except Exception:
			classes.update(["sidewalk", "road"])
	except Exception:
		pass
	if not classes:
		return ["gerobak", "sidewalk", "road"]
	return sorted(list(classes))

# verify and download weights
def verify_and_download_weights(model_b_path: Path):
	if not model_b_path.exists() and model_b_path == DEFAULT_MODEL_B:
		st.info("Pre-trained Model B (yolov11x-seg.pt) not found. Downloading fallback YOLO11n-seg for environment maps...")
		try:
			from ultralytics import YOLO
			tmp = YOLO("yolo11n-seg.pt")
			if Path("yolo11n-seg.pt").exists():
				shutil.move("yolo11n-seg.pt", str(model_b_path))
			st.success("Fallback environment model downloaded successfully!")
		except Exception as e:
			st.error(f"Failed to download pre-trained environment model: {e}")
			return False
	return True
@st.cache_resource

# load perception model
def load_perception_model(model_a: Path, model_b: Path, conf: float, device: str | None) -> DualModelPerception:
	return DualModelPerception(
		gerobak_weights=model_a,
		env_weights=model_b,
		gerobak_conf=conf,
		env_conf=conf,
		device=device,
	)

# draw detection overlays
def draw_detection_overlays(
	frame: np.ndarray,
	gerobak_masks: list[tuple[np.ndarray, str, float]],
	sidewalk_masks: list[tuple[np.ndarray, str, float]],
	keypoints: list[cv2.KeyPoint],
	src_points: np.ndarray,
	show_fast: bool = True,
	show_masks: bool = True,
	show_outlines: bool = True,
	show_polygons: bool = True,
	mask_alpha: float = 0.45,
) -> np.ndarray:
	overlay = frame.copy()

	if show_masks:
		for mask, _, _ in sidewalk_masks:
			overlay[mask > 0] = (0, 220, 0)
		for mask, _, _ in gerobak_masks:
			overlay[mask > 0] = (0, 0, 220)

	blended = cv2.addWeighted(overlay, mask_alpha, frame, 1.0 - mask_alpha, 0)

	for mask, label, score in gerobak_masks:
		if show_outlines:
			contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cv2.drawContours(blended, contours, -1, (0, 0, 255), 2)
			if len(contours) > 0:
				M = cv2.moments(contours[0])
				if M["m00"] != 0:
					cX = int(M["m10"] / M["m00"])
					cY = int(M["m01"] / M["m00"])
					text = f"{label} {score:.2f}"
					cv2.putText(blended, text, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	for mask, label, score in sidewalk_masks:
		if show_outlines:
			contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)
			if len(contours) > 0:
				M = cv2.moments(contours[0])
				if M["m00"] != 0:
					cX = int(M["m10"] / M["m00"])
					cY = int(M["m01"] / M["m00"])
					text = f"{label} {score:.2f}"
					cv2.putText(blended, text, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	if show_polygons:
		pts = src_points.astype(np.int32).reshape((-1, 1, 2))
		cv2.polylines(blended, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
		for idx, pt in enumerate(src_points):
			cv2.circle(blended, (int(pt[0]), int(pt[1])), 6, (255, 255, 0), -1)
			cv2.putText(blended, str(idx+1), (int(pt[0]) - 10, int(pt[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

	if show_fast and keypoints:
		for kp in keypoints:
			cv2.circle(blended, (int(kp.pt[0]), int(kp.pt[1])), 3, (0, 255, 255), -1)

	return blended

# draw bev map
def draw_bev_map(
	bev_sidewalk: np.ndarray,
	bev_gerobak: np.ndarray,
	bev_width: int,
	bev_height: int,
) -> np.ndarray:
	bev_map = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
	true_sidewalk = cv2.bitwise_or(bev_sidewalk, bev_gerobak)
	bev_map[true_sidewalk > 0] = (15, 100, 15)
	bev_map[bev_gerobak > 0] = (15, 15, 200)

	for y in range(0, bev_height, 100):
		cv2.line(bev_map, (0, y), (bev_width, y), (40, 50, 70), 1)
	return bev_map

# bgr to hex
def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
	return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"

# render legend
def render_legend(show_fast: bool = True) -> str:
	html = '<div class="metric-card">'
	legend_items = [
		("Sidewalk Area", "#00DC00"),
		("Gerobak Obstacle", "#DC0000"),
	]

	for label, color in legend_items:
		html += (
			f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
			f'<div style="width:14px;height:14px;background:{color};border-radius:3px;"></div>'
			f'<span style="font-size:0.9rem;">{label}</span>'
			f'</div>'
		)

	if show_fast:
		html += (
			'<div style="display:flex;align-items:center;gap:8px;margin-top:8px;border-top:1px solid var(--border);padding-top:8px;">'
			'<div style="width:14px;height:14px;background:#00C8FF;border-radius:3px;"></div>'
			'<span style="font-size:0.9rem;">FAST Features (Interest points)</span>'
			'</div>'
		)
	html += '</div>'

	return html

# process frame
def process_frame(
	frame: np.ndarray,
	perception_model: DualModelPerception,
	selected_classes: list[str],
	homography: np.ndarray,
	ratio: float,
	apply_fast: bool,
	fast_threshold: int,
	fast_min_count: int,
	src_points: np.ndarray,
	show_masks: bool,
	show_outlines: bool,
	show_polygons: bool,
	mask_alpha: float,
	bev_width: int = 400,
	bev_height: int = 600,
) -> dict:
	height, width = frame.shape[:2]
	output = perception_model.predict(frame)
	cleaned_gerobak: list[tuple[np.ndarray, str, float]] = []
	cleaned_sidewalk: list[tuple[np.ndarray, str, float]] = []
	all_fast_keypoints: list[cv2.KeyPoint] = []
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	sel_normalized = {c.lower().strip() for c in selected_classes}

	for det in output.gerobak:
		c_name = det.class_name.lower().strip()
		if c_name not in sel_normalized and "gerobak" not in sel_normalized:
			continue
		cleaned = clean_mask(det.mask)
		if c_name in ["sidewalk", "trotoar", "pavement"]:
			cleaned_sidewalk.append((cleaned, det.class_name, det.score))
			continue
		if apply_fast:
			verified = verify_gerobak_mask(gray, cleaned, threshold=fast_threshold, min_keypoints=fast_min_count)
			if not verified:
				continue
			kps = detect_fast_keypoints(gray, cleaned, threshold=fast_threshold)
			all_fast_keypoints.extend(kps)
		cleaned_gerobak.append((cleaned, det.class_name, det.score))

	for det in output.sidewalk:
		cleaned_sidewalk.append((clean_mask(det.mask), det.class_name, det.score))
	union_sidewalk = np.zeros((height, width), dtype=np.uint8)

	for mask_tuple in cleaned_sidewalk:
		mask = mask_tuple[0]
		if mask.shape[:2] != (height, width):
			mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
		union_sidewalk = cv2.bitwise_or(union_sidewalk, mask.astype(np.uint8))
	union_gerobak = np.zeros((height, width), dtype=np.uint8)

	for mask_tuple in cleaned_gerobak:
		mask = mask_tuple[0]
		if mask.shape[:2] != (height, width):
			mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
		union_gerobak = cv2.bitwise_or(union_gerobak, mask.astype(np.uint8))
	roi_mask = np.zeros((height, width), dtype=np.uint8)
	pts = src_points.astype(np.int32).reshape((-1, 1, 2))
	cv2.fillPoly(roi_mask, [pts], 255)
	masked_gerobak = []

	for mask, label, score in cleaned_gerobak:
		if mask.shape[:2] != (height, width):
			mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
		masked_m = cv2.bitwise_and(mask, mask, mask=roi_mask)
		if cv2.countNonZero(masked_m) > 0:
			masked_gerobak.append((masked_m, label, score))
	cleaned_gerobak = masked_gerobak
	masked_sidewalk = []

	for mask, label, score in cleaned_sidewalk:
		if mask.shape[:2] != (height, width):
			mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
		masked_m = cv2.bitwise_and(mask, mask, mask=roi_mask)
		if cv2.countNonZero(masked_m) > 0:
			masked_sidewalk.append((masked_m, label, score))
	cleaned_sidewalk = masked_sidewalk
	union_sidewalk = np.zeros((height, width), dtype=np.uint8)

	for mask_tuple in cleaned_sidewalk:
		mask = mask_tuple[0]
		union_sidewalk = cv2.bitwise_or(union_sidewalk, mask.astype(np.uint8))
	union_gerobak = np.zeros((height, width), dtype=np.uint8)

	for mask_tuple in cleaned_gerobak:
		mask = mask_tuple[0]
		union_gerobak = cv2.bitwise_or(union_gerobak, mask.astype(np.uint8))

	bev_sidewalk = warp_mask(union_sidewalk, homography, (bev_width, bev_height))
	bev_gerobak = warp_mask(union_gerobak, homography, (bev_width, bev_height))
	blocked_bev = bev_gerobak
	true_sidewalk_bev = cv2.bitwise_or(bev_sidewalk, bev_gerobak)
	total_sidewalk_area = compute_area_m2(true_sidewalk_bev, ratio)
	blocked_area = compute_area_m2(blocked_bev, ratio)
	clear_area = max(0.0, total_sidewalk_area - blocked_area)
	accessibility = compute_accessibility(clear_area, total_sidewalk_area)
	overlay_frame = draw_detection_overlays(
		frame=frame,
		gerobak_masks=cleaned_gerobak,
		sidewalk_masks=cleaned_sidewalk,
		keypoints=all_fast_keypoints,
		src_points=src_points,
		show_fast=apply_fast,
		show_masks=show_masks,
		show_outlines=show_outlines,
		show_polygons=show_polygons,
		mask_alpha=mask_alpha,
	)

	bev_map = draw_bev_map(bev_sidewalk, bev_gerobak, bev_width, bev_height)
	return {
		"overlay_frame": overlay_frame,
		"bev_map": bev_map,
		"total_sidewalk_area": total_sidewalk_area,
		"blocked_area": blocked_area,
		"clear_area": clear_area,
		"accessibility": accessibility,
		"gerobak_count": len(cleaned_gerobak),
		"fast_count": len(all_fast_keypoints),
	}

# run stream
def run_stream(
	video_source: int | str,
	perception_model: DualModelPerception,
	selected_classes: list[str],
	homography: np.ndarray,
	ratio: float,
	apply_fast: bool,
	fast_threshold: int,
	fast_min_count: int,
	src_points: np.ndarray,
	show_masks: bool,
	show_outlines: bool,
	show_polygons: bool,
	mask_alpha: float,
	frame_skip: int,
	metric_placeholders: tuple[st.delta_generator.DeltaGenerator, ...],
	combined_placeholder: st.delta_generator.DeltaGenerator,
	bev_placeholder: st.delta_generator.DeltaGenerator,
	status_placeholder: st.delta_generator.DeltaGenerator,
) -> None:
	cap = cv2.VideoCapture(video_source)

	if not cap.isOpened():
		st.error(f"Could not open video source: {video_source}")
		st.session_state["running"] = False
		return

	frame_index = st.session_state.get("video_frame_index", 0)
	if frame_index > 0 and isinstance(video_source, str):
		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
	last_time = None

	while st.session_state.get("running", False):
		if st.session_state.get("paused", False):
			break
		ok, frame = cap.read()

		if not ok:
			st.session_state["video_frame_index"] = 0
			st.session_state["running"] = False
			break
		frame_index += 1

		if frame_index % frame_skip != 0:
			continue
		h, w = frame.shape[:2]

		if st.session_state["frame_width"] != w or st.session_state["frame_height"] != h:
			st.session_state["frame_width"] = w
			st.session_state["frame_height"] = h
			st.session_state["video_frame_index"] = frame_index
			st.rerun()
		res = process_frame(
			frame=frame,
			perception_model=perception_model,
			selected_classes=selected_classes,
			homography=homography,
			ratio=ratio,
			apply_fast=apply_fast,
			fast_threshold=fast_threshold,
			fast_min_count=fast_min_count,
			src_points=src_points,
			show_masks=show_masks,
			show_outlines=show_outlines,
			show_polygons=show_polygons,
			mask_alpha=mask_alpha,
		)
		now = time.perf_counter()
		fps = 0.0 if last_time is None else 1.0 / max(now - last_time, 1e-6)
		last_time = now
		m1, m2, m3, m4, m5 = metric_placeholders
		a_color = get_accessibility_color(res["accessibility"])
		a_rating = get_accessibility_rating(res["accessibility"])
		m1.markdown(
			f'<div class="metric-card">'
			f'<div class="label">Accessibility</div>'
			f'<div class="value" style="color:{a_color};">{res["accessibility"]:.1f}%</div>'
			f'<div class="sub">{a_rating}</div>'
			f'</div>',
			unsafe_allow_html=True,
		)
		m2.markdown(metric_card("Obstacle Area", f"{res['blocked_area']:.2f} m²", f"{res['gerobak_count']} Gerobak(s)"), unsafe_allow_html=True)
		m3.markdown(metric_card("Sidewalk Area", f"{res['total_sidewalk_area']:.2f} m²", f"Clear: {res['clear_area']:.2f} m²"), unsafe_allow_html=True)
		m4.markdown(metric_card("FPS", f"{fps:.1f}", f"Processing skip: {frame_skip}x"), unsafe_allow_html=True)
		m5.markdown(metric_card("FAST Points", f"{res['fast_count']}", "Inside Cart Detections"), unsafe_allow_html=True)
		status_placeholder.markdown(
			'<div class="metric-card">'
			'<div class="label">Status</div>'
			'<div class="value" style="font-size:1.1rem;color:var(--green);">Running Auditor</div>'
			'</div>',
			unsafe_allow_html=True,
		)
		overlay_rgb = cv2.cvtColor(res["overlay_frame"], cv2.COLOR_BGR2RGB)
		bev_rgb = cv2.cvtColor(res["bev_map"], cv2.COLOR_BGR2RGB)
		combined_placeholder.image(overlay_rgb, caption="Perspective View + Overlays", use_container_width=True)
		bev_placeholder.image(bev_rgb, caption="Sidewalk Bird's Eye View (BEV)", use_container_width=True)
		st.session_state["last_results"] = {
			"accessibility": res["accessibility"],
			"blocked_area": res["blocked_area"],
			"gerobak_count": res["gerobak_count"],
			"total_sidewalk_area": res["total_sidewalk_area"],
			"clear_area": res["clear_area"],
			"fps": fps,
			"fast_count": res["fast_count"],
			"overlay_rgb": overlay_rgb,
			"bev_rgb": bev_rgb,
		}
		st.session_state["video_frame_index"] = frame_index
		time.sleep(0.01)
	cap.release()

	if not st.session_state.get("paused", False):
		st.session_state["running"] = False
		st.session_state["video_frame_index"] = 0
		status_placeholder.markdown(
			'<div class="metric-card">'
			'<div class="label">Status</div>'
			'<div class="value" style="font-size:1.1rem;">Stopped</div>'
			'</div>',
			unsafe_allow_html=True,
		)

st.markdown(
	'<div style="margin-bottom:1.5rem">'
	'<h1 style="font-size:2.4rem;font-weight:600;margin:0;color:var(--text-main);">'
	'NGALANGIN: Navigasi & Analisis Luas Area Gangguan Infrastruktur</h1>'
	'</div>',
	unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Analysis Dashboard", "Model Evaluation"])
with tab1:
	m1, m2, m3, m4, m5 = st.columns(5)
	metric_placeholders = (m1.empty(), m2.empty(), m3.empty(), m4.empty(), m5.empty())

	for metric in metric_placeholders:
		metric.markdown(metric_card("Metric", "--"), unsafe_allow_html=True)
	st.markdown("")
	frame_col1, frame_col2 = st.columns([1.2, 0.8])

	with frame_col1:
		combined_placeholder = st.empty()
		combined_placeholder.markdown(
			'<div class="metric-card" style="text-align:center;padding:5rem;color:var(--text-muted);">'
			'Perspective Frame View'
			'</div>',
			unsafe_allow_html=True,
		)

	with frame_col2:
		bev_placeholder = st.empty()
		bev_placeholder.markdown(
			'<div class="metric-card" style="text-align:center;padding:5rem;color:var(--text-muted);">'
			'Bird\'s Eye View Map'
			'</div>',
			unsafe_allow_html=True,
		)
	error_placeholder = st.empty()

class DummyPlaceholder:
	# markdown
	def markdown(self, *args, **kwargs):
		pass
	# image
	def image(self, *args, **kwargs):
		pass
	# empty
	def empty(self, *args, **kwargs):
		return self
status_placeholder = DummyPlaceholder()

with st.sidebar:
	st.markdown('<div class="section-heading" style="margin-top:0;">Control Panel</div>', unsafe_allow_html=True)
	col_btn1, col_btn2, col_btn3 = st.columns(3)
	is_paused = st.session_state.get("paused", False)
	is_running = st.session_state.get("running", False)

	with col_btn1:
		start_label = "▶ Resume" if (is_running and is_paused) else "▶ Start"
		st.button(
			start_label,
			use_container_width=True,
			type="primary",
			on_click=click_start_resume,
			disabled=is_running and not is_paused,
			key="btn_start_resume"
		)

	with col_btn2:
		st.button(
			"⏸ Pause",
			use_container_width=True,
			disabled=not is_running or is_paused,
			on_click=click_pause,
			key="btn_pause"
		)

	with col_btn3:
		st.button(
			"⏹ Stop",
			use_container_width=True,
			disabled=not is_running,
			on_click=click_stop,
			key="btn_stop"
		)

	with st.expander("Source Input Settings", expanded=True):
		source_type = st.selectbox(
			"Input Source Type",
			["File (Image/Video)", "Webcam"],
			key="source_type_select",
		)
		uploaded = None
		image_bytes: bytes | None = None
		video_source: int | str | None = None
		if source_type == "File (Image/Video)":
			uploaded = st.file_uploader(
				"Choose sidewalk image or video file",
				type=["png", "jpg", "jpeg", "bmp", "mp4", "avi", "mov", "mkv"],
				key="file_upload",
			)
			if uploaded:
				suffix = Path(uploaded.name).suffix.lower()
				if suffix in [".png", ".jpg", ".jpeg", ".bmp"]:
					uploaded.seek(0)
					image_bytes = uploaded.read()
				else:
					tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
					uploaded.seek(0)
					tmp.write(uploaded.read())
					tmp.close()
					video_source = tmp.name
		else:
			video_source = 0

	with st.expander("Model & Class Settings", expanded=True):
		available_models = get_available_models()
		model_a_options = list(available_models.keys())
		default_a_key = "yolov11_obstacle_model/weights/best.pt"
		default_a_idx = 0

		if default_a_key in model_a_options:
			default_a_idx = model_a_options.index(default_a_key)
		selected_model_a = st.selectbox(
			"Model A (Gerobak Obstacle)",
			options=model_a_options,
			index=default_a_idx,
			key="model_a_select",
		)
		weights_a_path = available_models[selected_model_a]
		model_b_options = list(available_models.keys())
		default_b_key = "yolov11x-seg.pt"
		default_b_idx = min(1, len(model_b_options) - 1) if model_b_options else 0

		if default_b_key in model_b_options:
			default_b_idx = model_b_options.index(default_b_key)
		selected_model_b = st.selectbox(
			"Model B (Environment Segmenter)",
			options=model_b_options,
			index=default_b_idx,
			key="model_b_select",
		)
		weights_b_path = available_models[selected_model_b]
		ensemble_classes = get_ensemble_classes(str(weights_a_path), str(weights_b_path))
		selected_classes = st.multiselect(
			"Classes to Detect & Audit",
			options=ensemble_classes,
			default=[c for c in ensemble_classes if c.lower() in ["gerobak", "sidewalk", "ngalangin", "motorcycle", "motor", "trotoar"]],
			key="classes_select_ensemble",
		)

	with st.expander("Calibration & Thresholds", expanded=False):
		conf_val = st.selectbox(
			"Model Confidence Threshold",
			options=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
			index=7,
			key="conf_threshold_select"
		)
		sidewalk_width_m = st.selectbox(
			"Real Sidewalk Width (meters)",
			options=[1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0],
			index=4,
			key="sidewalk_width_select"
		)
		roi_preset = st.selectbox(
			"ROI Presets (Perspective Transform)",
			options=["Default (Centered)", "Wide View", "Narrow View", "Custom Coordinates"],
			index=0,
			key="roi_preset_select"
		)
		w_val = st.session_state["frame_width"]
		h_val = st.session_state["frame_height"]

		if roi_preset == "Default (Centered)":
			x_tl, y_tl = int(w_val * 0.25), int(h_val * 0.55)
			x_bl, y_bl = int(w_val * 0.05), int(h_val * 0.95)
			x_tr, y_tr = int(w_val * 0.75), int(h_val * 0.55)
			x_br, y_br = int(w_val * 0.95), int(h_val * 0.95)
		elif roi_preset == "Wide View":
			x_tl, y_tl = int(w_val * 0.15), int(h_val * 0.50)
			x_bl, y_bl = int(w_val * 0.01), int(h_val * 0.98)
			x_tr, y_tr = int(w_val * 0.85), int(h_val * 0.50)
			x_br, y_br = int(w_val * 0.99), int(h_val * 0.98)
		elif roi_preset == "Narrow View":
			x_tl, y_tl = int(w_val * 0.35), int(h_val * 0.60)
			x_bl, y_bl = int(w_val * 0.15), int(h_val * 0.92)
			x_tr, y_tr = int(w_val * 0.65), int(h_val * 0.60)
			x_br, y_br = int(w_val * 0.85), int(h_val * 0.92)
		else:
			st.markdown("##### Custom ROI Coordinates")
			# parse pt
			def parse_pt(pt_str, default_val):
				try:
					parts = pt_str.replace("(", "").replace(")", "").split(",")
					if len(parts) == 2:
						return int(parts[0].strip()), int(parts[1].strip())
				except Exception:
					pass
				return default_val
			c1, c2 = st.columns(2)

			with c1:
				pt1_input = st.text_input("Top-Left (X, Y)", value=f"{int(w_val * 0.25)}, {int(h_val * 0.55)}", key="roi_pt1")
				pt4_input = st.text_input("Bottom-Left (X, Y)", value=f"{int(w_val * 0.05)}, {int(h_val * 0.95)}", key="roi_pt4")
			with c2:
				pt2_input = st.text_input("Top-Right (X, Y)", value=f"{int(w_val * 0.75)}, {int(h_val * 0.55)}", key="roi_pt2")
				pt3_input = st.text_input("Bottom-Right (X, Y)", value=f"{int(w_val * 0.95)}, {int(h_val * 0.95)}", key="roi_pt3")
			x_tl, y_tl = parse_pt(pt1_input, (int(w_val * 0.25), int(h_val * 0.55)))
			x_tr, y_tr = parse_pt(pt2_input, (int(w_val * 0.75), int(h_val * 0.55)))
			x_br, y_br = parse_pt(pt3_input, (int(w_val * 0.95), int(h_val * 0.95)))
			x_bl, y_bl = parse_pt(pt4_input, (int(w_val * 0.05), int(h_val * 0.95)))
		src_pts = np.array(
			[[x_tl, y_tl], [x_tr, y_tr], [x_br, y_br], [x_bl, y_bl]],
			dtype=np.float32,
		)
		bev_width = 400
		bev_height = 600
		dst_pts = np.array(
			[[0, 0], [bev_width, 0], [bev_width, bev_height], [0, bev_height]],
			dtype=np.float32,
		)
		H_matrix = compute_homography(src_pts, dst_pts)
		ratio = pixel_to_meter_ratio(bev_width, sidewalk_width_m)

	with st.expander("Advanced Settings & Styling", expanded=False):
		st.markdown('<div class="section-heading" style="margin-top:0;">FAST Point Verification</div>', unsafe_allow_html=True)
		apply_fast = st.checkbox("Enable FAST Verification", value=True, key="apply_fast_auditor")
		fast_threshold = st.slider("FAST Threshold", 1, 100, 25, key="fast_thresh_auditor")
		fast_min_count = st.slider("FAST Min Keypoints", 1, 50, 12, key="fast_min_auditor")
		frame_skip = st.slider("Frame Skip Rate", 1, 10, 2, key="frame_skip_auditor")
		st.markdown('<div class="section-heading">Visual Styling</div>', unsafe_allow_html=True)
		show_masks = st.checkbox("Show Mask Overlays", value=True, key="show_masks_check")
		show_outlines = st.checkbox("Show Mask Outlines", value=True, key="show_outlines_check")
		show_polygons = st.checkbox("Show Bounding Polygons", value=True, key="show_polygons_check")
		mask_alpha = st.slider("Mask Alpha (Opacity)", 0.0, 1.0, 0.45, step=0.05, key="mask_alpha_slider")
	st.markdown('<div class="section-heading">Legend</div>', unsafe_allow_html=True)
	legend_placeholder = st.empty()
	legend_placeholder.markdown(render_legend(show_fast=apply_fast), unsafe_allow_html=True)
current_source_key = f"{source_type}_{uploaded.name if uploaded else ''}"

if st.session_state.get("last_source_key") != current_source_key:
	st.session_state["last_source_key"] = current_source_key
	st.session_state["running"] = False
	st.session_state["paused"] = False
	st.session_state["video_frame_index"] = 0

	if "last_results" in st.session_state:
		del st.session_state["last_results"]

if st.session_state.get("running", False):
	if st.session_state.get("paused", False):
		if "last_results" in st.session_state:
			res = st.session_state["last_results"]
			m1, m2, m3, m4, m5 = metric_placeholders
			a_color = get_accessibility_color(res["accessibility"])
			a_rating = get_accessibility_rating(res["accessibility"])
			m1.markdown(
				f'<div class="metric-card">'
				f'<div class="label">Accessibility</div>'
				f'<div class="value" style="color:{a_color};">{res["accessibility"]:.1f}%</div>'
				f'<div class="sub">{a_rating}</div>'
				f'</div>',
				unsafe_allow_html=True,
			)
			m2.markdown(metric_card("Obstacle Area", f"{res['blocked_area']:.2f} m²", f"{res['gerobak_count']} Gerobak(s)"), unsafe_allow_html=True)
			m3.markdown(metric_card("Sidewalk Area", f"{res['total_sidewalk_area']:.2f} m²", f"Clear: {res['clear_area']:.2f} m²"), unsafe_allow_html=True)
			fps_val = res.get("fps")
			fps_str = "--" if fps_val is None else f"{fps_val:.1f}"
			fps_sub = "Single image analysis" if fps_val is None else f"Processing skip: {frame_skip}x"
			m4.markdown(metric_card("FPS", fps_str, fps_sub), unsafe_allow_html=True)
			m5.markdown(metric_card("FAST Points", f"{res['fast_count']}", "Inside Cart Detections"), unsafe_allow_html=True)
			status_placeholder.markdown(
				'<div class="metric-card">'
				'<div class="label">Status</div>'
				'<div class="value" style="font-size:1.1rem;color:var(--yellow);">Paused</div>'
				'</div>',
				unsafe_allow_html=True,
			)
			combined_placeholder.image(res["overlay_rgb"], caption="Perspective View + Overlays (Paused)", use_container_width=True)
			bev_placeholder.image(res["bev_rgb"], caption="Sidewalk Bird's Eye View (BEV) (Paused)", use_container_width=True)
	else:
		if not verify_and_download_weights(weights_b_path):
			st.session_state["running"] = False
		else:
			try:
				model = load_perception_model(weights_a_path, weights_b_path, conf=conf_val, device=None)
			except Exception as err:
				error_placeholder.error(f"Failed to initialize models: {err}")
				st.session_state["running"] = False
				st.stop()
			if source_type == "File (Image/Video)" and not video_source:
				if not image_bytes:
					error_placeholder.warning("Please upload an image or video before starting.")
					st.session_state["running"] = False
				else:
					image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
					if image is None:
						error_placeholder.error("Failed to load/decode image.")
						st.session_state["running"] = False
					else:
						status_placeholder.markdown(
							'<div class="metric-card">'
							'<div class="label">Status</div>'
							'<div class="value" style="font-size:1.1rem;color:var(--green);">Running Auditor</div>'
							'</div>',
							unsafe_allow_html=True,
						)
						h, w = image.shape[:2]
						if st.session_state["frame_width"] != w or st.session_state["frame_height"] != h:
							st.session_state["frame_width"] = w
							st.session_state["frame_height"] = h
							st.rerun()
						res = process_frame(
							frame=image,
							perception_model=model,
							selected_classes=selected_classes,
							homography=H_matrix,
							ratio=ratio,
							apply_fast=apply_fast,
							fast_threshold=fast_threshold,
							fast_min_count=fast_min_count,
							src_points=src_pts,
							show_masks=show_masks,
							show_outlines=show_outlines,
							show_polygons=show_polygons,
							mask_alpha=mask_alpha,
						)
						a_color = get_accessibility_color(res["accessibility"])
						a_rating = get_accessibility_rating(res["accessibility"])
						m1, m2, m3, m4, m5 = metric_placeholders
						m1.markdown(
							f'<div class="metric-card">'
							f'<div class="label">Accessibility</div>'
							f'<div class="value" style="color:{a_color};">{res["accessibility"]:.1f}%</div>'
							f'<div class="sub">{a_rating}</div>'
							f'</div>',
							unsafe_allow_html=True,
						)
						m2.markdown(metric_card("Obstacle Area", f"{res['blocked_area']:.2f} m²", f"{res['gerobak_count']} Gerobak(s)"), unsafe_allow_html=True)
						m3.markdown(metric_card("Sidewalk Area", f"{res['total_sidewalk_area']:.2f} m²", f"Clear: {res['clear_area']:.2f} m²"), unsafe_allow_html=True)
						m4.markdown(metric_card("FPS", "--", "Single image analysis"), unsafe_allow_html=True)
						m5.markdown(metric_card("FAST Points", f"{res['fast_count']}", "Inside Cart Detections"), unsafe_allow_html=True)
						overlay_rgb = cv2.cvtColor(res["overlay_frame"], cv2.COLOR_BGR2RGB)
						bev_rgb = cv2.cvtColor(res["bev_map"], cv2.COLOR_BGR2RGB)
						combined_placeholder.image(overlay_rgb, caption="Perspective View + Overlays", use_container_width=True)
						bev_placeholder.image(bev_rgb, caption="Sidewalk Bird's Eye View (BEV)", use_container_width=True)
						st.session_state["last_results"] = {
							"accessibility": res["accessibility"],
							"blocked_area": res["blocked_area"],
							"gerobak_count": res["gerobak_count"],
							"total_sidewalk_area": res["total_sidewalk_area"],
							"clear_area": res["clear_area"],
							"fps": None,
							"fast_count": res["fast_count"],
							"overlay_rgb": overlay_rgb,
							"bev_rgb": bev_rgb,
						}
						st.session_state["running"] = False
						status_placeholder.markdown(
							'<div class="metric-card">'
							'<div class="label">Status</div>'
							'<div class="value" style="font-size:1.1rem;">Stopped</div>'
							'</div>',
							unsafe_allow_html=True,
						)
			elif not video_source and video_source != 0:
				error_placeholder.warning("Please choose a video source before running.")
				st.session_state["running"] = False
			else:
				run_stream(
					video_source=video_source,
					perception_model=model,
					selected_classes=selected_classes,
					homography=H_matrix,
					ratio=ratio,
					apply_fast=apply_fast,
					fast_threshold=fast_threshold,
					fast_min_count=fast_min_count,
					src_points=src_pts,
					show_masks=show_masks,
					show_outlines=show_outlines,
					show_polygons=show_polygons,
					mask_alpha=mask_alpha,
					frame_skip=frame_skip,
					metric_placeholders=metric_placeholders,
					combined_placeholder=combined_placeholder,
					bev_placeholder=bev_placeholder,
					status_placeholder=status_placeholder,
				)
else:
	if "last_results" in st.session_state:
		res = st.session_state["last_results"]
		m1, m2, m3, m4, m5 = metric_placeholders
		a_color = get_accessibility_color(res["accessibility"])
		a_rating = get_accessibility_rating(res["accessibility"])
		m1.markdown(
			f'<div class="metric-card">'
			f'<div class="label">Accessibility</div>'
			f'<div class="value" style="color:{a_color};">{res["accessibility"]:.1f}%</div>'
			f'<div class="sub">{a_rating}</div>'
			f'</div>',
			unsafe_allow_html=True,
		)
		m2.markdown(metric_card("Obstacle Area", f"{res['blocked_area']:.2f} m²", f"{res['gerobak_count']} Gerobak(s)"), unsafe_allow_html=True)
		m3.markdown(metric_card("Sidewalk Area", f"{res['total_sidewalk_area']:.2f} m²", f"Clear: {res['clear_area']:.2f} m²"), unsafe_allow_html=True)
		fps_val = res.get("fps")
		fps_str = "--" if fps_val is None else f"{fps_val:.1f}"
		fps_sub = "Single image analysis" if fps_val is None else f"Processing skip: {frame_skip}x"
		m4.markdown(metric_card("FPS", fps_str, fps_sub), unsafe_allow_html=True)
		m5.markdown(metric_card("FAST Points", f"{res['fast_count']}", "Inside Cart Detections"), unsafe_allow_html=True)
		status_placeholder.markdown(
			'<div class="metric-card">'
			'<div class="label">Status</div>'
			'<div class="value" style="font-size:1.1rem;">Stopped</div>'
			'</div>',
			unsafe_allow_html=True,
		)
		combined_placeholder.image(res["overlay_rgb"], caption="Perspective View + Overlays", use_container_width=True)
		bev_placeholder.image(res["bev_rgb"], caption="Sidewalk Bird's Eye View (BEV)", use_container_width=True)
	else:
		preview_image = None
		if source_type == "File (Image/Video)" and uploaded:
			try:
				suffix = Path(uploaded.name).suffix.lower()
				if suffix in [".png", ".jpg", ".jpeg", ".bmp"]:
					uploaded.seek(0)
					preview_image = cv2.imdecode(np.frombuffer(uploaded.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
				elif video_source:
					cap = cv2.VideoCapture(video_source)
					if cap.isOpened():
						ok, preview_image = cap.read()
					cap.release()
			except Exception:
				pass

		if preview_image is not None:
			h, w = preview_image.shape[:2]
			if st.session_state["frame_width"] != w or st.session_state["frame_height"] != h:
				st.session_state["frame_width"] = w
				st.session_state["frame_height"] = h
				st.rerun()
			preview_blended = preview_image.copy()
			pts = src_pts.astype(np.int32).reshape((-1, 1, 2))
			cv2.polylines(preview_blended, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
			for idx, pt in enumerate(src_pts):
				cv2.circle(preview_blended, (int(pt[0]), int(pt[1])), 6, (255, 255, 0), -1)
				cv2.putText(preview_blended, str(idx+1), (int(pt[0]) - 10, int(pt[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
			preview_rgb = cv2.cvtColor(preview_blended, cv2.COLOR_BGR2RGB)
			combined_placeholder.image(preview_rgb, caption="ROI Calibration Preview (Analysis Idle)", use_container_width=True)
		else:
			combined_placeholder.markdown(
				'<div class="metric-card" style="text-align:center;padding:5rem;color:var(--text-muted);">'
				'Select a video/image source to begin accessibility analysis.'
				'</div>',
				unsafe_allow_html=True,
			)

# load eval metrics
def load_eval_metrics() -> dict:
	csv_path = MODEL_DIR / "yolov11_obstacle_model" / "results.csv"
	metrics = {
		"box_precision": 0.0, "box_recall": 0.0, "box_map50": 0.0, "box_map95": 0.0, "box_f1": 0.0,
		"mask_precision": 0.0, "mask_recall": 0.0, "mask_map50": 0.0, "mask_map95": 0.0, "mask_f1": 0.0,
		"epochs": 0
	}
	if csv_path.exists():
		try:
			with open(csv_path, "r") as f:
				lines = f.readlines()
			if len(lines) > 1:
				headers = [h.strip() for h in lines[0].split(",")]
				last_vals = [v.strip() for v in lines[-1].split(",")]
				row = dict(zip(headers, last_vals))
				metrics["epochs"] = len(lines) - 1
				metrics["box_precision"] = float(row.get("metrics/precision(B)", 0.0))
				metrics["box_recall"] = float(row.get("metrics/recall(B)", 0.0))
				metrics["box_map50"] = float(row.get("metrics/mAP50(B)", 0.0))
				metrics["box_map95"] = float(row.get("metrics/mAP50-95(B)", 0.0))
				metrics["mask_precision"] = float(row.get("metrics/precision(M)", 0.0))
				metrics["mask_recall"] = float(row.get("metrics/recall(M)", 0.0))
				metrics["mask_map50"] = float(row.get("metrics/mAP50(M)", 0.0))
				metrics["mask_map95"] = float(row.get("metrics/mAP50-95(M)", 0.0))
				p_b, r_b = metrics["box_precision"], metrics["box_recall"]
				if (p_b + r_b) > 0:
					metrics["box_f1"] = 2 * (p_b * r_b) / (p_b + r_b)
				p_m, r_m = metrics["mask_precision"], metrics["mask_recall"]
				if (p_m + r_m) > 0:
					metrics["mask_f1"] = 2 * (p_m * r_m) / (p_m + r_m)
		except Exception:
			pass
	return metrics
with tab2:
	st.markdown('<h3 style="color:var(--text-main);">Model Performance & Evaluation</h3>', unsafe_allow_html=True)
	col_info1, col_info2 = st.columns(2)

	with col_info1:
		st.markdown(
			'<div class="metric-card" style="height:175px;">'
			'<div class="label" style="color:var(--primary);font-weight:600;">Model A (Obstacle Segmenter)</div>'
			'<div class="value" style="font-size:1.3rem;margin-bottom:0.4rem;">YOLOv11-seg (Obstacle Model)</div>'
			'<div class="sub" style="font-size:0.85rem;line-height:1.4;">'
			'• <b>Weights:</b> <code>yolov11_obstacle_model/weights/best.pt</code><br>'
			'• <b>Target Class:</b> Gerobak (sidewalk carts)<br>'
			'• <b>Details:</b> Fine-tuned for detecting mobile sidewalk vendor obstacles.'
			'</div>'
			'</div>',
			unsafe_allow_html=True,
		)

	with col_info2:
		st.markdown(
			'<div class="metric-card" style="height:175px;">'
			'<div class="label" style="color:var(--green);font-weight:600;">Model B (Environment Segmenter)</div>'
			'<div class="value" style="font-size:1.3rem;margin-bottom:0.4rem;">YOLOv11x-seg</div>'
			'<div class="sub" style="font-size:0.85rem;line-height:1.4;">'
			'• <b>Weights:</b> <code>yolov11x-seg.pt</code><br>'
			'• <b>Target Class:</b> Sidewalk and road environments<br>'
			'• <b>Details:</b> Extracts contextual pavement boundaries and road limits.'
			'</div>'
			'</div>',
			unsafe_allow_html=True,
		)
	st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
	st.markdown('<h4 style="color:var(--text-main);">Model A Key Evaluation Metrics (Last Epoch)</h4>', unsafe_allow_html=True)
	eval_metrics = load_eval_metrics()
	col_m1, col_m2, col_m3, col_m4 = st.columns(4)

	with col_m1:
		st.markdown(
			f'<div class="metric-card">'
			f'<div class="label">mAP50 (Segmentation)</div>'
			f'<div class="value" style="color:var(--primary);">{eval_metrics["mask_map50"]*100:.1f}%</div>'
			f'<div class="sub">IoU Threshold: 0.50</div>'
			f'</div>',
			unsafe_allow_html=True,
		)

	with col_m2:
		st.markdown(
			f'<div class="metric-card">'
			f'<div class="label">mAP50-95 (mIoU / Seg)</div>'
			f'<div class="value" style="color:var(--green);">{eval_metrics["mask_map95"]*100:.1f}%</div>'
			f'<div class="sub">Mean Seg IoU 0.50-0.95</div>'
			f'</div>',
			unsafe_allow_html=True,
		)

	with col_m3:
		st.markdown(
			f'<div class="metric-card">'
			f'<div class="label">F1-Score (Segmentation)</div>'
			f'<div class="value" style="color:var(--yellow);">{eval_metrics["mask_f1"]*100:.1f}%</div>'
			f'<div class="sub">Harmonic mean of P & R</div>'
			f'</div>',
			unsafe_allow_html=True,
		)

	with col_m4:
		st.markdown(
			f'<div class="metric-card">'
			f'<div class="label">Precision / Recall</div>'
			f'<div class="value" style="font-size:1.3rem;line-height:2.2rem;">{eval_metrics["mask_precision"]*100:.1f}% / {eval_metrics["mask_recall"]*100:.1f}%</div>'
			f'<div class="sub">Total Epochs: {eval_metrics["epochs"]}</div>'
			f'</div>',
			unsafe_allow_html=True,
		)
	st.markdown("<br>", unsafe_allow_html=True)
	st.markdown("##### Detailed Metrics Comparison")
	st.markdown(
		f"""
		<table style="width:100%; border-collapse: collapse; margin-top: 1rem; border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden;">
			<thead>
				<tr style="background-color: var(--surface); border-bottom: 2px solid var(--border);">
					<th style="padding: 12px; text-align: left; color: var(--text-main); font-weight: 600;">Metric Type</th>
					<th style="padding: 12px; text-align: center; color: var(--text-main); font-weight: 600;">Precision</th>
					<th style="padding: 12px; text-align: center; color: var(--text-main); font-weight: 600;">Recall</th>
					<th style="padding: 12px; text-align: center; color: var(--text-main); font-weight: 600;">F1-Score</th>
					<th style="padding: 12px; text-align: center; color: var(--text-main); font-weight: 600;">mAP50</th>
					<th style="padding: 12px; text-align: center; color: var(--text-main); font-weight: 600;">mAP50-95 (mIoU)</th>
				</tr>
			</thead>
			<tbody>
				<tr style="border-bottom: 1px solid var(--border); background-color: var(--surface-2);">
					<td style="padding: 12px; color: var(--text-main); font-weight: 500;">Object Detection (Box)</td>
					<td style="padding: 12px; text-align: center; color: var(--text-main);">{eval_metrics["box_precision"]*100:.2f}%</td>
					<td style="padding: 12px; text-align: center; color: var(--text-main);">{eval_metrics["box_recall"]*100:.2f}%</td>
					<td style="padding: 12px; text-align: center; color: var(--text-main);">{eval_metrics["box_f1"]*100:.2f}%</td>
					<td style="padding: 12px; text-align: center; color: var(--text-main);">{eval_metrics["box_map50"]*100:.2f}%</td>
					<td style="padding: 12px; text-align: center; color: var(--text-main);">{eval_metrics["box_map95"]*100:.2f}%</td>
				</tr>
				<tr style="background-color: var(--surface-2);">
					<td style="padding: 12px; color: var(--text-main); font-weight: 500;">Segmentation (Mask)</td>
					<td style="padding: 12px; text-align: center; color: var(--text-main);">{eval_metrics["mask_precision"]*100:.2f}%</td>
					<td style="padding: 12px; text-align: center; color: var(--text-main);">{eval_metrics["mask_recall"]*100:.2f}%</td>
					<td style="padding: 12px; text-align: center; color: var(--text-main);">{eval_metrics["mask_f1"]*100:.2f}%</td>
					<td style="padding: 12px; text-align: center; color: var(--text-main);">{eval_metrics["mask_map50"]*100:.2f}%</td>
					<td style="padding: 12px; text-align: center; color: var(--text-main);">{eval_metrics["mask_map95"]*100:.2f}%</td>
				</tr>
			</tbody>
		</table>
		""",
		unsafe_allow_html=True
	)
	st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
	eval_dir = MODEL_DIR / "yolov11_obstacle_model"
	eval_subtab1, eval_subtab2, eval_subtab3, eval_subtab4 = st.tabs([
		"Training Progress & Loss (Model A)",
		"Confusion Matrix (Model A)",
		"PR & F1 Curves (Model A)",
		"Label Analysis (Model A)"
	])

	with eval_subtab1:
		st.markdown("#### Training Results and Loss Curves (Model A)")
		results_png = eval_dir / "results.png"
		if results_png.exists():
			st.image(str(results_png), caption="YOLOv11 training losses and metrics over epochs (Model A)", use_container_width=True)
		else:
			st.warning("Training results image (results.png) not found.")

	with eval_subtab2:
		st.markdown("#### Confusion Matrix (Model A)")
		col_cm1, col_cm2 = st.columns(2)
		with col_cm1:
			cm_norm = eval_dir / "confusion_matrix_normalized.png"
			if cm_norm.exists():
				st.image(str(cm_norm), caption="Normalized Confusion Matrix (Model A)", use_container_width=True)
		with col_cm2:
			cm_raw = eval_dir / "confusion_matrix.png"
			if cm_raw.exists():
				st.image(str(cm_raw), caption="Raw Confusion Matrix (Model A)", use_container_width=True)

	with eval_subtab3:
		st.markdown("#### Precision-Recall & F1-Confidence Curves (Model A)")
		st.markdown("##### Mask-level Segmentation Metrics (Model A)")
		col_mask1, col_mask2 = st.columns(2)
		with col_mask1:
			mask_pr = eval_dir / "MaskPR_curve.png"
			if mask_pr.exists():
				st.image(str(mask_pr), caption="Mask Precision-Recall Curve (Model A)", use_container_width=True)
		with col_mask2:
			mask_f1 = eval_dir / "MaskF1_curve.png"
			if mask_f1.exists():
				st.image(str(mask_f1), caption="Mask F1-Confidence Curve (Model A)", use_container_width=True)
		st.markdown("---")
		st.markdown("##### Box-level Detection Metrics (Model A)")
		col_box1, col_box2 = st.columns(2)
		with col_box1:
			box_pr = eval_dir / "BoxPR_curve.png"
			if box_pr.exists():
				st.image(str(box_pr), caption="Box Precision-Recall Curve (Model A)", use_container_width=True)
		with col_box2:
			box_f1 = eval_dir / "BoxF1_curve.png"
			if box_f1.exists():
				st.image(str(box_f1), caption="Box F1-Confidence Curve (Model A)", use_container_width=True)

	with eval_subtab4:
		st.markdown("#### Labels and Predictions Analysis (Model A)")
		col_lbl1, col_lbl2 = st.columns(2)
		with col_lbl1:
			labels_jpg = eval_dir / "labels.jpg"
			if labels_jpg.exists():
				st.image(str(labels_jpg), caption="Dataset Label Distributions", use_container_width=True)
		with col_lbl2:
			val_pred = eval_dir / "val_batch0_pred.jpg"
			if val_pred.exists():
				st.image(str(val_pred), caption="Validation Batch Predictions Sample", use_container_width=True)
