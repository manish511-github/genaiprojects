import cv2
import os
import numpy as np
from typing import TypedDict, List, Dict, Any, Union, Optional
import logging
from langgraph.graph import StateGraph, START, END
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

from dotenv import load_dotenv
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# State definition
class AgentState(TypedDict):
    video_dir: str
    query: Union[str, np.ndarray]
    is_image_query: bool
    video_paths: List[str]
    video_frames: Dict[str, List[Dict[str, Any]]]
    frame_descriptions: Dict[str, List[Dict[str, Any]]]
    search_results: List[Dict[str, Any]]
    metadata: Dict[str, Dict[str, Any]]
    faiss_index: Any
    id_to_frame: Dict[int, tuple]

def detect_scenes(video_path: str, max_frames: int =8) -> List[int]:
    """
    Detects scene changes in a video and returns the frame indices where scenes start.

    Parameters:
        video_path (str): Path to the input video file.
        max_frames (int): Maximum number of scene start frames to return. Defaults to 8.

    Returns:
        List[int]: A list of frame indices where scenes start. 
                   If no scenes are detected, returns [0].
                   In case of error, also returns [0].

    Notes:
        - Uses PySceneDetect's ContentDetector with a threshold of 30.0.
        - Automatically downscales video to improve performance.
        - Releases video resources after processing.
    """
    try:
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scenes = scene_manager.get_scene_list()
        frame_indices = [scene[0].get_frames() for scene in scenes[:max_frames]]
        video_manager.release()
        return frame_indices if frame_indices else [0]

    except Exception as e:
        logger.error(f"Error detecting scenes in {video_path}: {str(e)}")
        return [0]

def list_videos(state : AgentState) -> AgentState:
    """
    Lists video files in the specified directory and extracts their metadata.

    Args:
        state (AgentState): Application state containing "video_dir".

    Returns:
        AgentState: Updated state with "video_paths" (list of video file paths) 
            and "metadata" (video details like duration, title, fps).

    Raises:
        Exception: Logs and re-raises any errors.

    Notes:
        - Supports .mp4, .avi, .mov, .mkv files.
        - Uses OpenCV for metadata extraction.
    """
    try:
        video_dir = state["video_dir"]
        video_extensions = (".mp4", ".avi", ".mov", ".mkv")
        video_paths = [
            os.path.join(video_dir,f) for f in os.listdir(video_dir)
            if f.lower().endswith(video_extensions)
        ]
        metadata = {}
        for path in video_paths:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                metadata[path] = {
                    "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                    "title": os.path.basename(path),
                    "fps": cap.get(cv2.CAP_PROP_FPS)
                }
                cap.release()
                state["video_paths"] = video_paths
        state["metadata"] = metadata
        logger.info(f"Found {len(video_paths)} videos")
        return state
    except Exception as e:
        logger.error(f"Error listing videos: {str(e)}")
        raise


def extract_keyframes(state: AgentState) -> AgentState:
    """
    Extracts keyframes from videos specified in the state and updates the state with the extracted frames.
    Args:
        state (AgentState): The current state containing video paths.
    Returns:
        AgentState: Updated state with extracted keyframes for each video.
    """
    video_paths = state["video_paths"]
    video_frames = {}

    for video_path in video_paths:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            frame_indices = detect_scenes(video_path)
            logger.info(f"Frame Indexeas " + str(frame_indices))
            frames = []
            for idx in frame_indices[:8]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append({"frame_idx": idx, "frame": frame})
            cap.release()
            video_frames[video_path] = frames
            logger.info(f"Extracted {len(frames)} keyframes from {video_path}")
        except Exception as e:
            logger.error(f"Error extracting keyframes from {video_path}: {str(e)}")
    
    state["video_frames"] = video_frames
    return state

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    
    graph.add_node("list_videos", list_videos)
    graph.add_node("extract_keyframes", extract_keyframes)
    graph.add_node("analyze_frames", lambda state: asyncio.run(analyze_frames(state)))
    # graph.add_node("match_query", match_query)
    
    graph.add_edge(START, "list_videos")
    graph.add_edge("list_videos", "extract_keyframes")
    graph.add_edge("extract_keyframes", "analyze_frames")
    # graph.add_edge("analyze_frames", "match_query")
    # graph.add_edge("match_query", END)
    graph.add_edge("analyze_frames", END)
    return graph.compile()

async def run_video_search_agent(video_dir: str, query: Union[str, np.ndarray], is_image_query: bool = False) -> Dict[str, Any]:
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Video Directory not found: {video_dir} ")
    
    initial_state = AgentState(
        video_dir=video_dir,
        query=query,
        is_image_query=is_image_query,
        video_paths=[],
        video_frames={},
        frame_descriptions={},
        search_results=[],
        metadata={},
        faiss_index=None,
        id_to_frame={}
    )

    try:
        graph = build_graph()
        final_state = await asyncio.get_event_loop().run_in_executor(None, lambda: graph.invoke(initial_state))
        return final_state
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    try:

        video_dir = "videos"
        query = "Dog running AND park OR cat"
        result = asyncio.run(run_video_search_agent(video_dir, query, is_image_query = False))
        import pprint
        pprint.pprint(result)
    except Exception as e:
        print(f"Error: {str(e)}")
