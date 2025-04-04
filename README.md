# Image Tampering Detection using CNN 🚀

This repository contains a **Convolutional Neural Network (CNN)** model designed to detect whether a images such as **bank receipt** has been tampered with by a **human** or an **AI**.

---

## **🚀 API Usage**

### **POST /predict**

- Upload an image file (`.jpg` or `.jpeg`).
- Returns **whether the image is tampered or original**.

---

## **🐳 Running with Docker**

To build and run the Docker container:

```bash
# Step 1: Build the Docker Image
docker build -t fastapi-cnn-app .

# Step 2: Run the Container
docker run -p 8000:8000 fastapi-cnn-app
```

```

The API will be available at: **http://127.0.0.1:8000**
```

---

## **💻 Local Setup**

To run the project without Docker, follow these steps:

```bash
# Step 1: Clone the Repository
git clone https://github.com/Dharansh-Neema/image-tampering-detection
cd image-tampering-detection

# Step 2: Install Dependencies
pip install -r requirements.txt

# Step 3: Run the FastAPI Server
uvicorn main:app --reload
```

Access the API at **http://127.0.0.1:8000**

---

## **📊 Training Performance**

Below are the **training vs validation curves** for accuracy and loss:

### **Accuracy (Training vs Validation)**

![Training vs Validation Accuracy](util/Acc_validation.png)

### **Epoch Training Progress**

![Epoch Training](util/Epoch_data.png)

---

### **📝 Author**

👤 **Dharansh Neema**  
🔗 [GitHub Profile](https://github.com/Dharansh-Neema)

---
