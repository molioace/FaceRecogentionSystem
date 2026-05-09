Real-Time Face Recognition System
A modular real-time face recognition system built using FastAPI, OpenCV, YOLOv8-Face, ArcFace embeddings, and FAISS vector similarity search.
The project supports real-time webcam recognition, face registration, unknown face detection, face alignment using facial landmarks, and high-speed embedding similarity search.

Features


Real-time face recognition


Face registration from webcam or uploaded image


Unknown face detection


Face alignment using facial landmarks


Fast similarity search using FAISS


REST API using FastAPI


OpenCV real-time visualization


Modular service-based architecture


ONNX Runtime accelerated inference



Recognition Pipeline
Input Image    ↓YOLOv8-Face Detection    ↓Face Cropping    ↓Facial Landmark Detection    ↓Face Alignment    ↓ArcFace Embedding Extraction    ↓FAISS Similarity Search    ↓Recognition Result

Tech Stack
Backend


FastAPI


Uvicorn


Computer Vision


OpenCV


NumPy


AI Inference


ONNX Runtime


AI Models


YOLOv8-Face


ArcFace (w600k_r50)


2D106 Landmark Detector


Similarity Search


FAISS



Project Structure
FaceRecogentionSystem/│├── app/│   ├── routers/│   │   └── recognition.py│   ││   ├── services/│   │   ├── face_detector.py│   │   ├── face_pipeline.py│   │   ├── face_aligner.py│   │   ├── face_embedder.py│   │   ├── landmark_detector.py│   │   └── faiss_service.py│   ││   ├── config.py│   └── main.py│├── indexes/├── models/├── uploads/│├── scripts/│   ├── demo_register.py│   ├── demo_recognition.py│   └── download_models.py│├── requirements.txt└── README.md

Installation
Clone Repository
git clone https://github.com/molioace/FaceRecogentionSystem.gitcd FaceRecogentionSystem

Create Virtual Environment
Windows
python -m venv .venv.venv\Scripts\activate
Linux / macOS
python -m venv .venvsource .venv/bin/activate

Install Dependencies
pip install -r requirements.txt

Download Models
Run:
python scripts/download_models.py
This downloads the required pretrained models into the models/ directory.

Run Backend API
python -m uvicorn app.main:app --host 127.0.0.1 --port 5454

Register New Faces
python scripts/demo_register.py
Supports:


Webcam capture


Manual image upload



Run Real-Time Recognition
python scripts/demo_recognition.py
The recognition demo:


Opens webcam stream


Sends frames to backend API


Draws green boxes for recognized users


Draws red boxes for unknown users


Displays identity and similarity score



API Endpoints
Health Check
GET /recognition/health

Register Face
POST /recognition/register
Parameters


user_id


image file



Recognize Face
POST /recognition/recognize
Returns


Bounding boxes


Identity


Similarity score


Match status



Example Response
{  "status": "success",  "faces_count": 1,  "results": [    {      "bbox": [120, 80, 310, 290],      "identity": "Mohammed",      "similarity": 0.82,      "matched": true    }  ]}

Design Notes


Embeddings are normalized before similarity search


FAISS IndexFlatIP is used for cosine similarity


Unknown faces are automatically labeled when similarity is below threshold


Face alignment improves embedding consistency and recognition accuracy



Future Improvements


GPU inference support


Face tracking


Anti-spoofing detection


WebSocket streaming


Docker deployment


Database integration


User management dashboard


Multi-camera support



License
This project is intended for educational, research, and portfolio purposes.
