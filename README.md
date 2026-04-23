# NeuroVision: Generative Predictive Coding for Autonomous Drone Navigation

NeuroVision is a research-grade, bio-inspired generative inference engine designed to minimize the Detection-Response Gap in autonomous drone navigation. Unlike traditional discriminative pipelines (such as YOLO or SSD) that react to sensory input, NeuroVision utilizes a Predictive Coding (PC) architecture to proactively "dream" environmental states. By implementing a novel T+10 Recursive Leap-Frog Extrapolation, the system compensates for CPU hardware latency, allowing the drone to anticipate environmental changes approximately 333 milliseconds into the future.

This project bridges the gap between biological predictive processing and robotics by treating the drone as an active inference engine. The core objective is to minimize "Surprise" (prediction error) between top-down visual hypotheses and bottom-up sensory input, enabling faster and more intelligent navigation decisions.

NeuroVision introduces anticipatory awareness by projecting a "Future Ghost" tensor, which enables proactive maneuverability and reduces dependency on real-time computation speed. The system performs unsupervised anomaly detection by learning directly from raw environmental context, eliminating the need for labeled bounding boxes. It also employs a dual-metric surprise quantification approach using a weighted combination of Mean Squared Error (MSE) and Structural Similarity Index (SSIM) to differentiate between normal motion and structural anomalies. The entire architecture is optimized for edge environments and is capable of running efficiently on standard CPU hardware using recursive temporal extrapolation techniques.

The system is built using a modern deep learning and computer vision stack. PyTorch is used for implementing neural architectures, specifically PredNet based on ConvLSTM for predictive coding. OpenCV (cv2) is used for real-time vision processing and frame handling. Scikit-Learn and Matplotlib are used for diagnostics and visualization, including confusion matrix generation. Data logging is handled using CSV, Datetime, and Python Collections for structured telemetry storage.

To set up the project, first clone the repository using:
git clone https://github.com/your-username/NeuroVision.git  
cd NeuroVision  

Next, create and activate a virtual environment:
python -m venv venv  

On Windows:
venv\Scripts\activate  

On Mac/Linux:
source venv/bin/activate  

Install dependencies using:
pip install -r requirements.txt  

To run the system, execute:
python navigator.py  

Upon execution, the system initially displays a confusion matrix plot representing baseline validation metrics. Closing this window initializes the live drone vision pipeline. The system then launches three real-time visualization windows: one showing the current drone camera feed (Drone Reality), another displaying the predicted future frame (AI Future Sight using T+10 projection), and a third showing the anomaly heatmap (Surprise Map). These outputs allow the operator to understand both present and anticipated environmental states.

The system can be stopped by pressing the 'q' key, after which it safely releases the camera resources and writes all collected telemetry data to a file named drone_research_data.csv for further analysis.

The project is organized as follows:

NeuroVision/  
├── weights/              # Pre-trained model weights (.pth files)  
├── pytorch-prednet/      # Core PredNet architecture implementation  
├── data_utils.py         # Data preprocessing utilities  
├── navigator.py          # Main execution script  
├── requirements.txt      # Dependency list  
├── .gitignore            # Git exclusion rules  
├── License.txt           # License details  
└── README.md             # Project documentation  

If you are using this framework for research or academic purposes, please cite the foundational work on predictive coding networks:

Lotter, W., Kreiman, G., & Cox, D. (2016).  
Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning.  
arXiv:1605.08104  

This project is licensed under the MIT License. Refer to the License.txt file for detailed terms and conditions.

NeuroVision fundamentally shifts autonomous navigation from reactive perception to predictive intelligence, allowing drones to anticipate and respond to environmental changes before they fully occur.
