import os
import urllib.request
import bz2
import numpy as np
from PIL import Image
import cv2
import dlib
from imutils import face_utils

# Ensure the landmark model is available
def ensure_landmark_model():
    model_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        print("Downloading the facial landmark model...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
        print("Extracting the model...")
        with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2") as bz2_file:
            with open(model_path, "wb") as f:
                f.write(bz2_file.read())
        print("Model downloaded and extracted.")
    return model_path

# Main face detection function
def detect_ai_generated_face(image_path):
    """
    Analyzes an image of a face for signs of AI generation.
    Args:
        image_path (str): Path to the image file.
    Returns:
        str: Analysis result.
    """
    try:
        if not os.path.splitext(image_path)[1]:
            image_path += ".jpg"  # Default to .jpg if no extension is provided
        image = Image.open(image_path)
        image_array = np.array(image)
    except Exception as e:
        return f"Error loading image: {e}"

    # Step 1: Symmetry Analysis
    def check_symmetry(image_array):
        """
        Calculates symmetry based on differences between key facial features.
        Args:
            image_array: The input image array.
        Returns:
            float: Symmetry score (lower is more symmetric).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        # Detect face and landmarks
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(ensure_landmark_model())
        faces = detector(gray)

        if len(faces) == 0:
            return 1.0  # High score if no face detected

        for face in faces:
            # Get facial landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Extract individual points for left and right features
            left_points = np.vstack([shape[36:42], shape[30:31], shape[48:51]])  # Left eye, nose, mouth
            right_points = np.vstack([shape[42:48], shape[30:31], shape[54:57]])  # Right eye, nose, mouth

            # Calculate the absolute differences
            diff = np.abs(left_points - right_points)
            symmetry_score = np.mean(diff) / gray.shape[1]  # Normalize by image width

            return symmetry_score

        return 1.0  # Default high score for no landmarks detected

    symmetry_score = check_symmetry(image_array)

    # Step 2: Eye Alignment
    def detect_landmark_alignment(image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(ensure_landmark_model())
        faces = detector(gray)
        if len(faces) == 0:
            return 1.0
        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            def eye_center(eye_points):
                return np.mean(eye_points, axis=0)
            left_eye_center = eye_center(left_eye)
            right_eye_center = eye_center(right_eye)
            horizontal_diff = abs(left_eye_center[0] - right_eye_center[0])
            vertical_diff = abs(left_eye_center[1] - right_eye_center[1])
            alignment_score = vertical_diff / max(horizontal_diff, 1)
            return alignment_score
        return 1.0

    eye_alignment_score = detect_landmark_alignment(image_array)

    # Step 3: Texture Artifact Detection
    def detect_texture_artifacts(image_array):
        blurred = cv2.GaussianBlur(image_array, (15, 15), 0)
        diff = cv2.absdiff(image_array, blurred)
        texture_score = np.sum(diff) / (image_array.shape[0] * image_array.shape[1] * 255)
        return texture_score

    texture_score = detect_texture_artifacts(image_array)

    # Debugging Information
    print(f"Symmetry Score: {symmetry_score:.4f}")
    print(f"Eye Alignment Score: {eye_alignment_score:.4f}")
    print(f"Texture Artifact Score: {texture_score:.4f}")

    # Adjusted thresholds for your dataset
    if symmetry_score > 0.1:
        print("Symmetry score exceeds human threshold.")
    if eye_alignment_score > 0.2:
        print("Eye alignment score exceeds human threshold.")
    if texture_score > 0.1:
        print("Texture artifact score exceeds human threshold.")

    # Weighted scoring system
    ai_likelihood = (
        (symmetry_score * 0.2) +
        (eye_alignment_score * 0.2) +
        (texture_score * 0.6)
    )

    print(ai_likelihood)

    # Adjusted overall threshold
    if ai_likelihood > 0.05:  # Adjust threshold based on observed scores
        return "AI-generated face detected!"
    else:
        return "This appears to be a real human face."

# Continuous Testing Loop
while True:
    image_path = input("Enter the name of the image file (or 'exit' to quit): ").strip()
    if image_path.lower() == 'exit':
        print("Exiting program. Goodbye!")
        break
    result = detect_ai_generated_face(image_path)
    print(result)
