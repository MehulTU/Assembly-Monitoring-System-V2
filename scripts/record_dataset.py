"""
record_dataset.py

Prototype V2 - Controlled Dataset Acquisition

Purpose:
    Record controlled assembly trials from a camera.

Outputs:
    1. Video file for every completed trial.
    2. CSV metadata entry for every completed trial.

Controls:
    R = Start recording
    S = Stop recording
    Q = Quit program
"""

from pathlib import Path
from datetime import datetime
import csv
import time

import cv2


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

REQUESTED_FPS = 30.0

VIDEO_CODEC = "mp4v"

WINDOW_NAME = "Prototype V2 - Dataset Acquisition"


# ============================================================
# PROJECT PATHS
# ============================================================

# Location of this Python file:
# SOFTWARE V2 AI PROTOTYPE/scripts/record_dataset.py

SCRIPT_DIR = Path(__file__).resolve().parent

# Parent folder:
# SOFTWARE V2 AI PROTOTYPE/

PROJECT_ROOT = SCRIPT_DIR.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"

RAW_DATA_DIR = DATASETS_DIR / "raw"

VIDEO_DIR = RAW_DATA_DIR / "videos"

METADATA_DIR = RAW_DATA_DIR / "metadata"

METADATA_FILE = METADATA_DIR / "trials.csv"


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

VIDEO_DIR.mkdir(parents=True, exist_ok=True)

METADATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CREATE METADATA CSV FILE IF IT DOES NOT EXIST
# ============================================================

CSV_COLUMNS = [
    "trial_id",
    "video_filename",
    "start_timestamp",
    "end_timestamp",
    "duration_seconds",
    "frame_count",
    "frame_width",
    "frame_height",
    "requested_fps",
    "camera_index",
]


if not METADATA_FILE.exists():

    with METADATA_FILE.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS,
        )

        writer.writeheader()


# ============================================================
# GENERATE UNIQUE TRIAL ID
# ============================================================

def generate_trial_id():
    """
    Generate a unique trial identifier using date and time.

    Example:
        T_20260708_183045
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    return f"T_{timestamp}"


# ============================================================
# SAVE TRIAL METADATA
# ============================================================

def save_trial_metadata(
    trial_id,
    video_filename,
    start_timestamp,
    end_timestamp,
    duration_seconds,
    frame_count,
    frame_width,
    frame_height,
    requested_fps,
    camera_index,
):
    """
    Append information about one completed trial to trials.csv.
    """

    metadata = {
        "trial_id": trial_id,
        "video_filename": video_filename,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "duration_seconds": round(duration_seconds, 3),
        "frame_count": frame_count,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "requested_fps": requested_fps,
        "camera_index": camera_index,
    }

    with METADATA_FILE.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS,
        )

        writer.writerow(metadata)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 65)
    print("PROTOTYPE V2 - CONTROLLED DATASET ACQUISITION")
    print("=" * 65)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Video folder : {VIDEO_DIR}")
    print(f"Metadata file: {METADATA_FILE}")

    print("-" * 65)

    print("CONTROLS")
    print("R = Start recording")
    print("S = Stop recording")
    print("Q = Quit")

    print("-" * 65)


    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )


    if not camera.isOpened():

        print(f"ERROR: Could not open camera index {CAMERA_INDEX}.")

        return


    # --------------------------------------------------------
    # REQUEST CAMERA SETTINGS
    # --------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        REQUESTED_FPS,
    )


    # --------------------------------------------------------
    # READ ACTUAL CAMERA SETTINGS
    # --------------------------------------------------------

    actual_width = int(
        camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    actual_height = int(
        camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    actual_fps = camera.get(
        cv2.CAP_PROP_FPS
    )


    print(
        f"Camera opened successfully: "
        f"{actual_width}x{actual_height} "
        f"at reported {actual_fps:.2f} FPS"
    )


    # --------------------------------------------------------
    # RECORDING STATE VARIABLES
    # --------------------------------------------------------

    is_recording = False

    video_writer = None

    trial_id = None

    video_filename = None

    recording_start_time = None

    start_timestamp = None

    recorded_frame_count = 0


    # --------------------------------------------------------
    # MAIN CAMERA LOOP
    # --------------------------------------------------------

    try:

        while True:

            success, frame = camera.read()


            # ------------------------------------------------
            # FRAME VALIDATION
            # ------------------------------------------------

            if not success or frame is None:

                print("WARNING: Invalid camera frame received.")

                continue


            # ------------------------------------------------
            # WRITE FRAME TO VIDEO
            # ------------------------------------------------

            if is_recording and video_writer is not None:

                video_writer.write(frame)

                recorded_frame_count += 1


            # ------------------------------------------------
            # CREATE DISPLAY FRAME
            # ------------------------------------------------

            display_frame = frame.copy()


            if is_recording:

                elapsed_time = (
                    time.perf_counter()
                    - recording_start_time
                )

                status_text = (
                    f"RECORDING | "
                    f"{trial_id} | "
                    f"{elapsed_time:.1f} s | "
                    f"{recorded_frame_count} frames"
                )

            else:

                status_text = (
                    "READY | "
                    "R = Record | "
                    "S = Stop | "
                    "Q = Quit"
                )


            cv2.putText(
                display_frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255) if is_recording else (0, 255, 0),
                2,
                cv2.LINE_AA,
            )


            # ------------------------------------------------
            # SHOW LIVE CAMERA
            # ------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                display_frame,
            )


            # ------------------------------------------------
            # READ KEYBOARD INPUT
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF


            # =================================================
            # R = START RECORDING
            # =================================================

            if key == ord("r"):

                if is_recording:

                    print(
                        "WARNING: A trial is already being recorded."
                    )

                    continue


                trial_id = generate_trial_id()

                video_filename = f"{trial_id}.mp4"

                video_path = VIDEO_DIR / video_filename


                fourcc = cv2.VideoWriter_fourcc(
                    *VIDEO_CODEC
                )


                video_writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    REQUESTED_FPS,
                    (actual_width, actual_height),
                )


                if not video_writer.isOpened():

                    print(
                        f"ERROR: Could not create video file: "
                        f"{video_path}"
                    )

                    video_writer = None

                    continue


                recording_start_time = time.perf_counter()

                start_timestamp = (
                    datetime.now().isoformat(
                        timespec="milliseconds"
                    )
                )

                recorded_frame_count = 0

                is_recording = True


                print()

                print("=" * 65)

                print(f"RECORDING STARTED: {trial_id}")

                print(f"Video: {video_path}")

                print("=" * 65)


            # =================================================
            # S = STOP RECORDING
            # =================================================

            elif key == ord("s"):

                if not is_recording:

                    print(
                        "WARNING: No active recording to stop."
                    )

                    continue


                end_time = time.perf_counter()

                end_timestamp = (
                    datetime.now().isoformat(
                        timespec="milliseconds"
                    )
                )

                duration_seconds = (
                    end_time
                    - recording_start_time
                )


                video_writer.release()

                video_writer = None

                is_recording = False


                save_trial_metadata(
                    trial_id=trial_id,
                    video_filename=video_filename,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    duration_seconds=duration_seconds,
                    frame_count=recorded_frame_count,
                    frame_width=actual_width,
                    frame_height=actual_height,
                    requested_fps=REQUESTED_FPS,
                    camera_index=CAMERA_INDEX,
                )


                print()

                print("=" * 65)

                print(f"RECORDING STOPPED: {trial_id}")

                print(
                    f"Duration: "
                    f"{duration_seconds:.3f} seconds"
                )

                print(
                    f"Frames recorded: "
                    f"{recorded_frame_count}"
                )

                print("Metadata saved successfully.")

                print("=" * 65)


                trial_id = None

                video_filename = None

                recording_start_time = None

                start_timestamp = None

                recorded_frame_count = 0


            # =================================================
            # Q = QUIT
            # =================================================

            elif key == ord("q"):

                print("Quit requested.")

                break


    # =========================================================
    # SAFE CLEANUP
    # =========================================================

    finally:

        if is_recording and video_writer is not None:

            end_time = time.perf_counter()

            end_timestamp = (
                datetime.now().isoformat(
                    timespec="milliseconds"
                )
            )

            duration_seconds = (
                end_time
                - recording_start_time
            )


            video_writer.release()


            save_trial_metadata(
                trial_id=trial_id,
                video_filename=video_filename,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                duration_seconds=duration_seconds,
                frame_count=recorded_frame_count,
                frame_width=actual_width,
                frame_height=actual_height,
                requested_fps=REQUESTED_FPS,
                camera_index=CAMERA_INDEX,
            )


            print(
                "Active recording was safely stopped "
                "and metadata was saved."
            )


        camera.release()

        cv2.destroyAllWindows()

        print("Camera released.")

        print("Program closed safely.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()