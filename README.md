# Multi-View 3D Reconstruction using Attention and Visual Odometry

This project implements a deep-learning–driven **multi-view 3D reconstruction** pipeline that eliminates the need for LiDAR or IMU sensors. By combining **visual odometry**, **MNASNet feature extraction**, **FPN multi-scale enhancement**, and a **novel attention-based unprojection mechanism**, the system predicts TSDF volumes and reconstructs 3D meshes purely from RGB images.

---

## 🏗 Architecture Overview

Below is the high-level architecture pipeline from input video to final 3D mesh:

![Architecture](architecture.jpeg)

---

## 🚀 Features

- Multi-view 3D reconstruction  
- Novel attention mechanism for weighted voxel fusion  
- Geometric + deep learning visual odometry  
- Lightweight MNASNet backbone  
- Feature Pyramid Networks (FPN) for multi-scale fusion  
- 3D refinement using Residual UNet  
- TSDF regression and mesh extraction  
- Compatible with ScanNet, KITTI, Tanks & Temples  

---

## 🎯 Objective

To build a deep-learning model that can reconstruct a 3D scene using **only RGB images**, replacing expensive depth sensors. The model learns to regress **TSDF values** for each voxel using camera intrinsics, estimated extrinsics, and multi-view feature projections.

---

## 📦 Dataset

### **ScanNet Dataset**

- RGB-D indoor scenes  
- LiDAR-based ground truth TSDF  
- 27 views per scene  
- Keyframe thresholds:
  - Rotation > 15°  
  - Translation > 10 cm  

| Split | Count |
|-------|--------|
| Train | 6000 |
| Validation | 1110 |
| Test | 630 |

---

## ⚙️ System Components

### **1. Keyframe Selection**
Frames chosen only when ego-motion exceeds:
- Rotation threshold  
- Translation threshold  

### **2. Visual Odometry**
Hybrid approach:
- Geometric method (Essential matrix + optical flow)
- DeepVO (LSTM + FlowNet)
- Combined for improved pose accuracy  

### **3. Feature Extraction**
- Backbone: **MNASNet** (pretrained on ImageNet)  
- Lightweight and mobile-efficient  

### **4. Feature Enhancement**
- **FPN** merges coarse & fine feature maps  
- Produces semantically rich high-resolution features  

### **5. Unprojection (Key Contribution)**
Two mechanisms:
1. **Conventional** additive feature fusion  
2. **Attention-based unprojection**, which applies learned weights before accumulation  

### **6. 3D Refinement**
- Residual UNet for TSDF refinement  
- Upsampling, skip connections, and spatial consistency  

---

## 🧪 Training Details

### **Loss Functions**
- **Binary Cross Entropy (BCE)** → Occupancy  
- **L1 Loss** → TSDF regression  

### **Optimizer**
- **Adam**, learning rate: 1e–3  

### **Performance Improvements**

| Metric | Before Attention | After Attention |
|--------|------------------|-----------------|
| L1 TSDF Loss | 1.23 | **0.93** |

---

## 📊 Results

Below are sample outputs of the predicted mesh compared with the LiDAR-based ground truth:

![Results](result.jpeg)

The attention-based model reconstructs more accurate geometry, especially in fine-detail regions.

---

## 🏁 Conclusion

This project demonstrates effective **sensor-free 3D reconstruction** using purely RGB images. The proposed attention mechanism significantly improves voxel fusion quality, and the hybrid visual odometry approach enhances camera pose estimation.

Future improvements:
- C++ accelerated attention for real-time use  
- Integration into SLAM pipelines  
- Lightweight 3D models for mobile deployment  

---

## 📄 License
This project is licensed under the **MIT License**.

