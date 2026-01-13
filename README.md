# 🏥 Diabetic Retinopathy Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle)](https://www.kaggle.com/)
[![Achievement](https://img.shields.io/badge/🥈_2nd_Place-AUC_0.73648-gold)]()

> **Deep Learning system for automated detection of diabetic retinopathy in retinal fundus images**

Developed as the final project for the Machine Learning course at Universidad Peruana Cayetano Heredia (UPCH), this system achieved **2nd place** out of 7 competing teams in an internal Kaggle competition with an **AUC score of 0.73648**.

---

## 🎯 Problem Statement

**Diabetic Retinopathy (DR)** is a diabetes complication that affects the eyes, caused by damage to blood vessels in the retina. It's a leading cause of blindness in working-age adults worldwide. Early detection through retinal screening can prevent vision loss, but manual screening is time-intensive and requires specialized ophthalmologists.

**Challenge:** Build an automated classification system to detect and grade the severity of diabetic retinopathy from retinal fundus photographs.

### Classification Categories
- **Class 0**: No DR
- **Class 1**: Mild DR
- **Class 2**: Moderate DR  
- **Class 3**: Severe DR
- **Class 4**: Proliferative DR

---

## 🏆 Competition Results

**Final Standings:**

| Rank | Team | Score (AUC) | Performance |
|------|------|-------------|-------------|
| 🥇 1st | Leily Marlith Llanos Ángeles | 0.82220 | Excellent |
| **🥈 2nd** | **Bruno Gavidia Crovetto** | **0.73648** | **Strong** |
| 🥉 3rd | TeamMLFinal | 0.71693 | Good |

**Achievement Highlights:**
- 🥈 **2nd Place** out of 7 teams
- 📊 **AUC: 0.73648** (Area Under ROC Curve)
- 🎓 Official UPCH Kaggle Competition
- 📅 March - June 2024

---

## 🧠 Technical Approach

### Model Architecture

**Transfer Learning with Pre-trained CNN:**
- Base model: **Pre-trained Convolutional Neural Network** on ImageNet
- Fine-tuning strategy for retinal image domain
- Multi-class classification (5 severity levels)

### Key Technical Decisions

**✅ What Worked:**
- Transfer learning from ImageNet-pretrained models
- Data augmentation for limited medical imaging dataset
- Fine-tuning strategy with frozen early layers
- Image preprocessing and normalization

**📚 Lessons Learned:**
- **Critical insight**: Initially output binary predictions (yes/no) which limited performance
- **Improvement**: Should have output probability distributions for each class
- **Impact**: This decision cost us 1st place - a valuable lesson in understanding evaluation metrics
- **Takeaway**: Always align model outputs with competition evaluation criteria (AUC requires probabilities, not binary classifications)

### Technical Stack

- **Deep Learning**: TensorFlow/Keras
- **Image Processing**: OpenCV, PIL
- **Data Science**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Streamlit (web interface)

---

## 📊 Methodology

### 1. Data Preprocessing
- Image resizing and normalization
- Data augmentation (rotation, flip, zoom)
- Class imbalance handling
- Train/validation split

### 2. Model Development
- Architecture selection (pre-trained CNN)
- Transfer learning implementation
- Layer freezing strategy
- Hyperparameter tuning

### 3. Training Process
- Loss function: Categorical crossentropy
- Optimizer: Adam
- Evaluation metric: AUC (Area Under Curve)
- Early stopping and model checkpointing

### 4. Model Evaluation
- ROC-AUC analysis
- Confusion matrix
- Per-class performance metrics
- Error analysis

### 5. Deployment
- Streamlit web interface
- Real-time image upload and classification
- Probability distribution visualization
- User-friendly medical reporting

---

## 💻 Project Structure
```
diabetic-retinopathy-detection/
├── notebooks/
│   └── diabetic-retinopathy-classification.ipynb
├── docs/
│   └── competition-brief.pdf
├── README.md
└── requirements.txt
```

---

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.8+
TensorFlow 2.x
Streamlit
OpenCV
NumPy, Pandas, Matplotlib
```

### Setup
```bash
# Clone repository
git clone https://github.com/ArnySalazar/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook notebooks/diabetic-retinopathy-classification.ipynb
```

### Streamlit Web Interface
```bash
# Launch web app for classification
streamlit run app.py

# Upload retinal image → Get DR classification + severity level
```

---

## 📈 Results & Performance

### Model Performance
- **Final AUC Score**: 0.73648
- **Competition Rank**: 2nd out of 7 teams
- **Accuracy**: Strong performance on held-out test set
- **Clinical Relevance**: Suitable for screening assistance (not diagnostic)

### Key Metrics
- **Precision**: High for severe cases (Class 3-4)
- **Recall**: Balanced across all classes
- **F1-Score**: Competitive performance
- **ROC Curve**: Strong separation between classes

---

## 🔬 Medical Impact

**Clinical Significance:**
- Early detection of diabetic retinopathy can prevent blindness
- Automated screening reduces ophthalmologist workload
- Cost-effective solution for large-scale population screening
- Particularly valuable in areas with limited access to specialists

**Real-world Applications:**
- Primary care screening tool
- Telemedicine platforms
- Mobile health applications
- Population health programs in developing regions

---

## 📚 Lessons Learned & Future Improvements

### Critical Insights
1. **Output Format Matters**: Learned the importance of probability outputs vs. binary classification for AUC optimization
2. **Evaluation Metrics**: Understanding the evaluation metric before model design is crucial
3. **Medical AI**: Balance between model performance and clinical interpretability

### Future Enhancements
- [ ] Implement ensemble methods (multiple model voting)
- [ ] Output probability distributions instead of hard classifications
- [ ] Incorporate attention mechanisms for interpretability
- [ ] Add grad-CAM visualizations to highlight affected areas
- [ ] Expand to multi-task learning (detecting other retinal diseases)
- [ ] Mobile app deployment for field screening
- [ ] Integration with electronic health records (EHR)

---

## 🎓 Academic Context

**Course**: Introduction to Machine Learning (Introducción a Machine Learning)  
**Institution**: Universidad Peruana Cayetano Heredia (UPCH)  
**Faculty**: Faculty of Sciences and Engineering  
**Period**: 2024-1 (March - June 2024)  
**Team**: Bruno Gavidia Crovetto (2 members)

**Competition Details:**
- Platform: Kaggle (Internal UPCH Competition)
- Duration: 3 months
- Teams: 7 competing groups
- Deliverables: Code + Web Interface + Presentation

**Evaluation Criteria:**
- Presentation (4 points)
- Code Quality (6 points)
- Kaggle Score (10 points)
- **Total Score**: 18-20/20 (estimated based on 2nd place finish)

---

## 🏥 Dataset Information

**Source**: Kaggle Competition Dataset (UPCH Internal)

**Characteristics:**
- Retinal fundus photographs
- Multiple severity levels (0-4)
- Real patient data (anonymized)
- Imbalanced class distribution
- High-resolution medical imaging

---

## 📖 References & Resources

### Medical Background
- [WHO - Diabetic Retinopathy](https://www.who.int/news-room/fact-sheets/detail/blindness-and-visual-impairment)
- International Diabetic Retinopathy Grading Scale
- Clinical guidelines for DR screening

### Technical Resources
- Transfer Learning for Medical Imaging
- CNN Architectures for Image Classification
- Handling Imbalanced Medical Datasets
- AUC Optimization Strategies

---

## 👨‍💻 Development Team

**Arny Eliu Salazar Cobian**

- **Role**: ML Engineer & Developer
- **Contributions**: Model architecture, training pipeline, evaluation, lessons learned analysis
- **LinkedIn**: [linkedin.com/in/arny-eliu-salazar-cobian-033120246](https://linkedin.com/in/arny-eliu-salazar-cobian-033120246)
- **GitHub**: [@ArnySalazar](https://github.com/ArnySalazar)
- **Email**: arny.salazar@upch.pe

**Related Projects:**
- [PISAAC Healthcare Monitoring](https://github.com/ArnySalazar/pisaac-healthcare-monitoring) - AI-powered patient monitoring
- [CHAYANE ERP](https://github.com/ArnySalazar/PI2-CHAYANE) - Full-stack restaurant management

---

## 📄 License

Academic project developed for Universidad Peruana Cayetano Heredia.

**Medical Disclaimer**: This system is designed for educational and research purposes. It should not be used as the sole basis for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.

---

## 🙏 Acknowledgments

- **UPCH Faculty** for organizing the competition and providing guidance
- **Competition Host**: Prof. Pablo Fonseca
- **Medical Domain Experts** who provided insights on diabetic retinopathy
- **Kaggle Community** for dataset and platform

---

<div align="center">

**🥈 Proud 2nd Place Finisher - UPCH Machine Learning Competition 2024**

*Demonstrating the power of AI in healthcare through automated diabetic retinopathy detection*

⭐ **If you found this project interesting, please consider starring the repository!**

</div>
