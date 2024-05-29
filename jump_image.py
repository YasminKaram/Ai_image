from flask import Flask, request, jsonify,Blueprint
import cv2
import numpy as np
import mediapipe as mp
import base64 

jump_image=Blueprint('jump_image',__name__)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils 

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    vec_ab = a - b
    vec_bc = c - b

    dot_product = np.dot(vec_ab, vec_bc)
    magnitude_ab = np.linalg.norm(vec_ab)
    magnitude_bc = np.linalg.norm(vec_bc)

    angle_radians = np.arccos(dot_product / (magnitude_ab * magnitude_bc))
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def detectPose(image):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))
    return output_image, landmarks

def classifyPose(landmarks, output_image):
    label = 'Unknown Pose'
    color = (0, 0, 255)
 
    counter = 0
  

    elbowL= [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]]
    shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]]
    #wristL = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]]
    cac1=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        ]

                # Calculate angle
    angleL = calculate_angle(*cac1)
                # Get coordinates
    elbowR= [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]]
    shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]]
    #wristR = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]]
    cac2=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
          ]

                # Calculate angle
    angleR = calculate_angle(*cac2)

                # Visualize angle
  

    if angleR and angleL > 130:
        label='Jump Jacks is true'

    
        
    if label != 'Unknown Pose':
        color = (0, 255, 0)  
        
    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    return output_image, label

@jump_image.route('/jumpImage', methods=['POST'])
def detect_pose():
    if 'photo' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['photo']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image, landmarks = detectPose(image)
    if landmarks:
        output_image, label = classifyPose(landmarks, output_image)
        _, img_encoded = cv2.imencode('.jpg', output_image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        cv2.imwrite('static/processed_image.jpg', output_image)
        image_url = request.url_root + 'static/processed_image.jpg'
        response = {"label": label, "image": img_base64}
        return jsonify(response)
    else:
        return jsonify({"error": "No pose detected"}), 400

