"""
visualize_process_video.py

HOW TO RUN THE FILE:
    python scripts\visualize_process_video.py  (TYPE THIS IN TERMINAL)

Prototype V2 - Final Process Visualization Video Generator

Purpose:
    Create a final demonstration video that combines the original
    experiment video with the outputs of the complete computer-vision
    and process-monitoring pipeline.

The visualization shows:

    1. YOLO object detections and bounding boxes.
    2. Object class names and confidence scores.
    3. Temporally stabilized object-presence states.
    4. Current assembly/process state.
    5. Current frame number and video time.
    6. Sequence-violation warning when the state machine reports one.
    7. Assembly-completion notification when the state machine reports
       a completion event.
    8. Running counts of process transitions, sequence violations,
       and assembly completion events.

Inputs:

    datasets/analytics/raw_detections.csv

    datasets/analytics/detection_timeline.csv

    datasets/analytics/state_timeline.csv

    datasets/analytics/event_log.csv

    datasets/raw/videos/<source_video>

Output:

    datasets/analytics/process_visualization.mp4

Important:
    This script does not perform YOLO inference again.

    It visualizes the results already produced by the previous stages
    of the pipeline.

    The visualization is intended for debugging, demonstration,
    validation, presentations, and thesis documentation.
"""

from pathlib import Path
import csv
from collections import defaultdict

import cv2


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYTICS_FOLDER = PROJECT_ROOT / "datasets" / "analytics"

RAW_VIDEO_FOLDER = PROJECT_ROOT / "datasets" / "raw" / "videos"

RAW_DETECTIONS_FILE = (
    ANALYTICS_FOLDER / "raw_detections.csv"
)

DETECTION_TIMELINE_FILE = (
    ANALYTICS_FOLDER / "detection_timeline.csv"
)

STATE_TIMELINE_FILE = (
    ANALYTICS_FOLDER / "state_timeline.csv"
)

EVENT_LOG_FILE = (
    ANALYTICS_FOLDER / "event_log.csv"
)

OUTPUT_VIDEO_FILE = (
    ANALYTICS_FOLDER / "process_visualization.mp4"
)


# ======================================================================
# VISUALIZATION SETTINGS
# ======================================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX

BOX_THICKNESS = 2

TEXT_THICKNESS = 2

SMALL_FONT_SCALE = 0.55

NORMAL_FONT_SCALE = 0.70

LARGE_FONT_SCALE = 0.90


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def print_header(title):

    print()

    print("=" * 75)

    print(title)

    print("=" * 75)


def text_to_bool(value):

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def safe_int(value, default=0):

    try:

        return int(float(value))

    except (ValueError, TypeError):

        return default


def safe_float(value, default=0.0):

    try:

        return float(value)

    except (ValueError, TypeError):

        return default


def find_column(fieldnames, candidates, description):

    for candidate in candidates:

        if candidate in fieldnames:

            return candidate

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {fieldnames}"
    )


def load_csv_rows(file_path):

    with open(
        file_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        rows = list(reader)

        fieldnames = reader.fieldnames or []

    return rows, fieldnames


def draw_text_with_background(
    frame,
    text,
    position,
    font_scale,
    text_color,
    background_color,
    thickness=2,
    padding=5,
):

    x, y = position

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        FONT,
        font_scale,
        thickness,
    )

    top_left = (
        x - padding,
        y - text_height - padding,
    )

    bottom_right = (
        x + text_width + padding,
        y + baseline + padding,
    )

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        background_color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        FONT,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


# ======================================================================
# MAIN PROCESS
# ======================================================================

def main():

    print_header(
        "PROTOTYPE V2 - FINAL PROCESS VISUALIZATION VIDEO"
    )

    print(f"Raw detections:\n{RAW_DETECTIONS_FILE}")

    print()

    print(f"Detection timeline:\n{DETECTION_TIMELINE_FILE}")

    print()

    print(f"State timeline:\n{STATE_TIMELINE_FILE}")

    print()

    print(f"Event log:\n{EVENT_LOG_FILE}")

    print()

    print(f"Output video:\n{OUTPUT_VIDEO_FILE}")


    # ------------------------------------------------------------------
    # VERIFY INPUT FILES
    # ------------------------------------------------------------------

    print_header("VERIFYING INPUT FILES")

    required_files = [
        RAW_DETECTIONS_FILE,
        DETECTION_TIMELINE_FILE,
        STATE_TIMELINE_FILE,
        EVENT_LOG_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

        print(f"Found: {file_path.name}")


    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------

    print_header("LOADING PIPELINE OUTPUTS")

    raw_detection_rows, raw_detection_fields = load_csv_rows(
        RAW_DETECTIONS_FILE
    )

    detection_timeline_rows, detection_timeline_fields = load_csv_rows(
        DETECTION_TIMELINE_FILE
    )

    state_rows, state_fields = load_csv_rows(
        STATE_TIMELINE_FILE
    )

    event_rows, event_fields = load_csv_rows(
        EVENT_LOG_FILE
    )

    if not state_rows:

        raise ValueError(
            "State timeline contains no rows."
        )

    print(
        f"Raw detection rows       : "
        f"{len(raw_detection_rows)}"
    )

    print(
        f"Detection timeline rows  : "
        f"{len(detection_timeline_rows)}"
    )

    print(
        f"State timeline rows      : "
        f"{len(state_rows)}"
    )

    print(
        f"Event rows               : "
        f"{len(event_rows)}"
    )


    # ------------------------------------------------------------------
    # FIND REQUIRED COLUMNS
    # ------------------------------------------------------------------

    detection_frame_column = find_column(
        raw_detection_fields,
        [
            "frame_number",
            "source_frame_number",
            "frame_index",
        ],
        "raw detection frame number",
    )

    class_name_column = find_column(
        raw_detection_fields,
        [
            "class_name",
            "object_class",
            "class",
        ],
        "class name",
    )

    confidence_column = find_column(
        raw_detection_fields,
        [
            "confidence",
            "confidence_score",
            "conf",
        ],
        "confidence",
    )

    x1_column = find_column(
        raw_detection_fields,
        ["x1", "bbox_x1"],
        "bounding-box x1",
    )

    y1_column = find_column(
        raw_detection_fields,
        ["y1", "bbox_y1"],
        "bounding-box y1",
    )

    x2_column = find_column(
        raw_detection_fields,
        ["x2", "bbox_x2"],
        "bounding-box x2",
    )

    y2_column = find_column(
        raw_detection_fields,
        ["y2", "bbox_y2"],
        "bounding-box y2",
    )

    timeline_frame_column = find_column(
        detection_timeline_fields,
        [
            "frame_number",
            "source_frame_number",
            "frame_index",
        ],
        "detection timeline frame number",
    )

    state_frame_column = find_column(
        state_fields,
        [
            "frame_number",
            "source_frame_number",
            "frame_index",
        ],
        "state timeline frame number",
    )

    state_name_column = find_column(
        state_fields,
        [
            "assembly_state",
            "process_state",
            "state",
        ],
        "assembly state",
    )

    event_frame_column = find_column(
        event_fields,
        [
            "frame_number",
            "source_frame_number",
            "frame_index",
        ],
        "event frame number",
    )


    # ------------------------------------------------------------------
    # SOURCE VIDEO
    # ------------------------------------------------------------------

    source_videos = {
        row["source_video"]
        for row in state_rows
    }

    if len(source_videos) != 1:

        raise ValueError(
            "Expected exactly one source video."
        )

    source_video_name = next(iter(source_videos))

    source_video_path = (
        RAW_VIDEO_FOLDER / source_video_name
    )

    if not source_video_path.exists():

        raise FileNotFoundError(
            f"Source video not found:\n{source_video_path}"
        )

    print()

    print(f"Source video:\n{source_video_path}")


    # ------------------------------------------------------------------
    # INDEX RAW DETECTIONS BY FRAME
    # ------------------------------------------------------------------

    detections_by_frame = defaultdict(list)

    for row in raw_detection_rows:

        frame_number = safe_int(
            row[detection_frame_column]
        )

        detections_by_frame[frame_number].append(row)


    # ------------------------------------------------------------------
    # INDEX DETECTION TIMELINE
    # ------------------------------------------------------------------

    detection_timeline_by_frame = {}

    for row in detection_timeline_rows:

        frame_number = safe_int(
            row[timeline_frame_column]
        )

        detection_timeline_by_frame[frame_number] = row


    # ------------------------------------------------------------------
    # INDEX STATE TIMELINE
    # ------------------------------------------------------------------

    state_by_frame = {}

    for row in state_rows:

        frame_number = safe_int(
            row[state_frame_column]
        )

        state_by_frame[frame_number] = row


    # ------------------------------------------------------------------
    # INDEX EVENTS BY FRAME
    # ------------------------------------------------------------------

    events_by_frame = defaultdict(list)

    for row in event_rows:

        frame_number = safe_int(
            row[event_frame_column]
        )

        events_by_frame[frame_number].append(row)


    # ------------------------------------------------------------------
    # OPEN VIDEO
    # ------------------------------------------------------------------

    print_header("OPENING SOURCE VIDEO")

    video_capture = cv2.VideoCapture(
        str(source_video_path)
    )

    if not video_capture.isOpened():

        raise RuntimeError(
            f"Could not open source video:\n{source_video_path}"
        )

    fps = video_capture.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_video_frames = int(
        video_capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(f"FPS          : {fps:.3f}")

    print(f"Resolution   : {width} x {height}")

    print(f"Video frames : {total_video_frames}")


    # ------------------------------------------------------------------
    # CREATE VIDEO WRITER
    # ------------------------------------------------------------------

    OUTPUT_VIDEO_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    video_writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_FILE),
        fourcc,
        fps,
        (width, height),
    )

    if not video_writer.isOpened():

        video_capture.release()

        raise RuntimeError(
            f"Could not create output video:\n{OUTPUT_VIDEO_FILE}"
        )


    # ------------------------------------------------------------------
    # VISUALIZATION PROCESS
    # ------------------------------------------------------------------

    print_header("GENERATING VISUALIZATION VIDEO")

    frame_number = 0

    processed_frames = 0

    transition_count = 0

    violation_count = 0

    completion_count = 0

    last_event_text = ""

    last_event_frame = -100000

    EVENT_DISPLAY_FRAMES = max(
        1,
        int(round(fps * 1.5)),
    )


    while True:

        read_success, frame = video_capture.read()

        if not read_success:

            break


        # --------------------------------------------------------------
        # DRAW YOLO DETECTIONS
        # --------------------------------------------------------------

        frame_detections = detections_by_frame.get(
            frame_number,
            [],
        )

        for detection in frame_detections:

            class_name = detection[class_name_column]

            confidence = safe_float(
                detection[confidence_column]
            )

            x1 = safe_int(
                detection[x1_column]
            )

            y1 = safe_int(
                detection[y1_column]
            )

            x2 = safe_int(
                detection[x2_column]
            )

            y2 = safe_int(
                detection[y2_column]
            )


            if class_name == "marker":

                box_color = (0, 255, 0)

            elif class_name == "power_adapter":

                box_color = (255, 200, 0)

            else:

                box_color = (255, 255, 255)


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                BOX_THICKNESS,
            )


            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )


            label_y = max(
                25,
                y1 - 8,
            )


            draw_text_with_background(
                frame=frame,
                text=label,
                position=(x1, label_y),
                font_scale=SMALL_FONT_SCALE,
                text_color=(0, 0, 0),
                background_color=box_color,
                thickness=1,
            )


        # --------------------------------------------------------------
        # GET TEMPORAL DETECTION STATE
        # --------------------------------------------------------------

        timeline_row = detection_timeline_by_frame.get(
            frame_number
        )


        if timeline_row is not None:

            stable_marker_present = text_to_bool(
                timeline_row.get(
                    "stable_marker_present",
                    False,
                )
            )

            stable_adapter_present = text_to_bool(
                timeline_row.get(
                    "stable_power_adapter_present",
                    False,
                )
            )

        else:

            stable_marker_present = False

            stable_adapter_present = False


        # --------------------------------------------------------------
        # GET ASSEMBLY STATE
        # --------------------------------------------------------------

        state_row = state_by_frame.get(
            frame_number
        )


        if state_row is not None:

            assembly_state = state_row[
                state_name_column
            ]

        else:

            assembly_state = "NO_STATE_DATA"


        # --------------------------------------------------------------
        # PROCESS EVENTS
        # --------------------------------------------------------------

        current_events = events_by_frame.get(
            frame_number,
            [],
        )


        for event in current_events:

            transition_count += 1


            if text_to_bool(
                event.get(
                    "sequence_violation",
                    False,
                )
            ):

                violation_count += 1


            if text_to_bool(
                event.get(
                    "assembly_completed",
                    False,
                )
            ):

                completion_count += 1


            last_event_text = event.get(
                "event_type",
                "UNKNOWN_EVENT",
            )

            last_event_frame = frame_number


        # --------------------------------------------------------------
        # DRAW TOP INFORMATION PANEL
        # --------------------------------------------------------------

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (width, 145),
            (0, 0, 0),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.60,
            frame,
            0.40,
            0,
            frame,
        )


        timestamp_seconds = (
            frame_number / fps
            if fps > 0
            else 0.0
        )


        cv2.putText(
            frame,
            f"Prototype V2 Process Monitoring",
            (20, 30),
            FONT,
            NORMAL_FONT_SCALE,
            (255, 255, 255),
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )


        cv2.putText(
            frame,
            (
                f"Frame: {frame_number}  "
                f"Time: {timestamp_seconds:.2f} s"
            ),
            (20, 60),
            FONT,
            SMALL_FONT_SCALE,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


        cv2.putText(
            frame,
            (
                f"Stable presence | "
                f"Marker: "
                f"{'YES' if stable_marker_present else 'NO'}  "
                f"Adapter: "
                f"{'YES' if stable_adapter_present else 'NO'}"
            ),
            (20, 90),
            FONT,
            SMALL_FONT_SCALE,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


        cv2.putText(
            frame,
            f"Assembly state: {assembly_state}",
            (20, 125),
            FONT,
            LARGE_FONT_SCALE,
            (255, 255, 255),
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )


        # --------------------------------------------------------------
        # DRAW RUNNING STATISTICS
        # --------------------------------------------------------------

        statistics_text = (
            f"Transitions: {transition_count} | "
            f"Violations: {violation_count} | "
            f"Completions: {completion_count}"
        )


        cv2.putText(
            frame,
            statistics_text,
            (20, height - 25),
            FONT,
            SMALL_FONT_SCALE,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


        # --------------------------------------------------------------
        # DRAW RECENT EVENT
        # --------------------------------------------------------------

        if (
            frame_number - last_event_frame
            <= EVENT_DISPLAY_FRAMES
        ):

            draw_text_with_background(
                frame=frame,
                text=f"EVENT: {last_event_text}",
                position=(
                    20,
                    height - 60,
                ),
                font_scale=NORMAL_FONT_SCALE,
                text_color=(255, 255, 255),
                background_color=(60, 60, 60),
                thickness=2,
            )


        # --------------------------------------------------------------
        # DRAW VIOLATION WARNING
        # --------------------------------------------------------------

        current_violation = any(

            text_to_bool(
                event.get(
                    "sequence_violation",
                    False,
                )
            )

            for event in current_events
        )


        if current_violation:

            draw_text_with_background(
                frame=frame,
                text="SEQUENCE VIOLATION",
                position=(
                    width - 320,
                    height - 60,
                ),
                font_scale=NORMAL_FONT_SCALE,
                text_color=(255, 255, 255),
                background_color=(0, 0, 255),
                thickness=2,
            )


        # --------------------------------------------------------------
        # DRAW COMPLETION NOTIFICATION
        # --------------------------------------------------------------

        current_completion = any(

            text_to_bool(
                event.get(
                    "assembly_completed",
                    False,
                )
            )

            for event in current_events
        )


        if current_completion:

            draw_text_with_background(
                frame=frame,
                text="ASSEMBLY COMPLETION EVENT",
                position=(
                    width - 410,
                    height - 100,
                ),
                font_scale=NORMAL_FONT_SCALE,
                text_color=(0, 0, 0),
                background_color=(0, 255, 255),
                thickness=2,
            )


        # --------------------------------------------------------------
        # WRITE OUTPUT FRAME
        # --------------------------------------------------------------

        video_writer.write(frame)

        processed_frames += 1

        frame_number += 1


        if processed_frames % 100 == 0:

            print(
                f"Processed frames: "
                f"{processed_frames}/{total_video_frames}"
            )


    # ------------------------------------------------------------------
    # RELEASE VIDEO RESOURCES
    # ------------------------------------------------------------------

    video_capture.release()

    video_writer.release()


    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    print_header("VISUALIZATION SUMMARY")

    print(f"Source video              : {source_video_name}")

    print(f"Processed frames          : {processed_frames}")

    print(f"State transitions shown   : {transition_count}")

    print(f"Sequence violations shown : {violation_count}")

    print(f"Completion events shown   : {completion_count}")

    print()

    print(f"Output video saved to:\n{OUTPUT_VIDEO_FILE}")


    print_header(
        "STATUS: FINAL PROCESS VISUALIZATION CREATED SUCCESSFULLY"
    )

    print(
        "Offline end-to-end Prototype V2 pipeline completed."
    )


if __name__ == "__main__":

    main()