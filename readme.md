3D Object Overlay on Video Frames:
This project processes video frames to overlay a 3D object on the detected mouth region of a person in each frame. It uses the `mediapipe` library for mouth detection, `open3d` for 3D object processing, and `pyrender` for rendering the 3D object onto the video frames.

Prerequisites:
Make sure you have the following libraries installed:
- opencv-python-headless
- mediapipe
- open3d
- pyrender
- trimesh
- numpy
- matplotlib

You can install them using pip:
pip install opencv-python-headless mediapipe open3d pyrender trimesh numpy matplotlib
or 
pip install -r requirements.txt

Project Structure:
- extract_frames.py: Extracts frames from the video.
- process_frame.py: Processes each frame to detect the mouth and overlay the 3D object.
- main.py: Main script to run the entire pipeline.
- video.mp4: Input video file.
- 3d_model.stl: 3D model file in STL format.
- frames/: Folder to store extracted frames.
- output/: Folder to store processed frames.
- output_video.mp4: Final output video file.

How to Run:
1) Ensure you have all the prerequisites installed.
2) Update the paths in the main script to point to your video file, 3D model file, and desired output directories.
3) Run the main script to extract frames, process each frame to overlay the 3D model, and compile the processed frames into a video.

About Functions:
- The extract_frames function extracts frames from the video at the specified FPS and stores them in the given frames_folder.
- The load_and_prepare_model function loads and scales the 3D model.
- The detect_mouth_position function detects the mouth position in a frame using mediapipe.
- The process_frame function overlays the 3D model on the detected mouth position in each frame.
- The main function orchestrates the entire process, from frame extraction to video creation.
