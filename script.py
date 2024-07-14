#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install opencv-python-headless mediapipe open3d pyrender')


# In[1]:


import os
import cv2
import numpy as np
import open3d as o3d
import pyrender
import trimesh
import matplotlib.pyplot as plt
import mediapipe as mp


# In[2]:


# Extract frames from the video
def extract_frames(video_path, frames_folder, desired_fps=30):
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
        
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_interval = int(fps / desired_fps)
    
    success, image = vidcap.read()
    count = 0
    frame_count = 0
    
    while success:
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(frames_folder, f"{count}.png")
            cv2.imwrite(frame_path, image)
            count += 1

        frame_count += 1
        success, image = vidcap.read()
    
    vidcap.release()
    print("Frames extracted successfully.")
    return width, height


# In[3]:


# Function to load and prepare the 3D model
def load_and_prepare_model(stl_file_path, scale_factor=0.005):
    mesh = o3d.io.read_triangle_mesh(stl_file_path)
    mesh.compute_vertex_normals()
    mesh.scale(scale_factor, center=mesh.get_center())
    vertices = np.asarray(mesh.vertices)
    vertices -= vertices.mean(axis=0)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.triangles))
    return trimesh_mesh


# In[4]:


def detect_mouth_position(rgb_image, width, height):
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5) as pose:
        result = pose.process(rgb_image)
    
    if not result.pose_landmarks:
        return None, None
    
    landmarks = result.pose_landmarks.landmark
    mouth_left_index = mp_pose.PoseLandmark.MOUTH_LEFT.value
    mouth_right_index = mp_pose.PoseLandmark.MOUTH_RIGHT.value
    
    mouth_left = landmarks[mouth_left_index]
    mouth_right = landmarks[mouth_right_index]
    
    mouth_left_img = np.array([mouth_left.x * width, mouth_left.y * height])
    mouth_right_img = np.array([mouth_right.x * width, mouth_right.y * height])
    
    return mouth_left_img, mouth_right_img


# In[5]:


def process_frame(image_path, trimesh_mesh, width, height):
    background_image_rgb = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if background_image_rgb is None:
        print(f"Error: Background image at {image_path} could not be loaded. Skipping.")
        return None
    
    mouth_left_img, mouth_right_img = detect_mouth_position(background_image_rgb, width, height)
    
    if mouth_left_img is None or mouth_right_img is None:
        print(f"No mouth detected in image {image_path}. Skipping.")
        return None
    
    mouth_midpoint_img = (mouth_left_img + mouth_right_img) / 2
    
    scene = pyrender.Scene()
    fov = 60
    aspect_ratio = width / height
    camera = pyrender.PerspectiveCamera(yfov=np.radians(fov), aspectRatio=aspect_ratio)
    
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    object_pose = np.eye(4)
    object_pose[0, 3] = (mouth_midpoint_img[0] / width) * 2 - 1
    object_pose[1, 3] = 1 - (mouth_midpoint_img[1] / height) * 2
    object_pose[2, 3] = 0.5
    
    render_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
    scene.add(render_mesh, pose=object_pose)
    
    light = pyrender.PointLight(color=np.ones(3), intensity=1.0)
    scene.add(light, pose=camera_pose)
    
    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, _ = r.render(scene)
    
    blended_image = cv2.addWeighted(background_image_rgb, 0.5, color, 0.5, 0)
    blended_image_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
    
    return blended_image_rgb


# In[6]:


def main(video_path, frames_folder, stl_path, output_folder, output_video_path, desired_fps=30):
    width, height = extract_frames(video_path, frames_folder, desired_fps)
    
    trimesh_mesh = load_and_prepare_model(stl_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_files = sorted([f for f in os.listdir(frames_folder) if os.path.isfile(os.path.join(frames_folder, f))], key=lambda x: int(os.path.splitext(x)[0]))
    
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), desired_fps, (width, height))
    
    for frame_file in frame_files:
        input_image_path = os.path.join(frames_folder, frame_file)
        output_image_path = os.path.join(output_folder, frame_file)
        
        frame = cv2.imread(input_image_path)
        if frame is None:
            print(f"Error: Could not load image {input_image_path}. Skipping.")
            continue
        
        processed_image = process_frame(input_image_path, trimesh_mesh, width, height)
        
        if processed_image is not None:
            cv2.imwrite(output_image_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            video_writer.write(cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            print(f"Processed image saved to {output_image_path}")
        else:
            print(f"Skipping image {input_image_path} due to processing errors.")
    
    video_writer.release()
    print(f"Video saved to {output_video_path}")


# In[7]:


video_path = "C:\\Users\\sheik\\OneDrive\\Desktop\\mirror_software\\01.mp4"
frames_folder = "C:\\Users\\sheik\\OneDrive\\Desktop\\mirror_software\\frames_new"
stl_path = "C:\\Users\\sheik\\OneDrive\\Desktop\\mirror_software\\11042024-caso barbara cano - yo-maxillary.stl"
output_folder = "C:\\Users\\sheik\\OneDrive\\Desktop\\mirror_software\\output"
output_video_path = "C:\\Users\\sheik\\OneDrive\\Desktop\\mirror_software\\output_video.mp4"

main(video_path, frames_folder, stl_path, output_folder, output_video_path)


# In[ ]:




