import os
import cv2
import numpy as np
from sklearn import svm, neighbors, tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from time import time

# Hàm trích xuất đặc trưng HOG từ ảnh
def extract_features(image):
    # Chuyển ảnh sang grayscale nếu cần thiết
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Đảm bảo ảnh có kích thước 128x128
    image = cv2.resize(image, (128, 128))
    
    # Kiểm tra định dạng ảnh trước khi tính toán HOG
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Khởi tạo bộ trích xuất HOG và trích xuất đặc trưng
    hog = cv2.HOGDescriptor()
    try:
        features = hog.compute(image).flatten()
        return features
    except cv2.error as e:
        print(f"Error computing HOG features for image: {e}")
        return None

# Hàm chuẩn bị dữ liệu từ thư mục ảnh
def prepare_data(image_folder):
    images = []
    labels = []
    
    # Duyệt qua các thư mục con (nhãn)
    for label in os.listdir(image_folder):
        label_folder = os.path.join(image_folder, label)
        
        # Kiểm tra nếu label_folder là thư mục
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                image = cv2.imread(img_path)
                
                # Kiểm tra nếu ảnh được tải thành công
                if image is not None:
                    features = extract_features(image)
                    
                    # Thêm đặc trưng vào images nếu trích xuất thành công
                    if features is not None:
                        images.append(features)
                        labels.append(label)
                else:
                    print(f"Image at {img_path} could not be loaded.")
    
    return np.array(images), np.array(labels)

# Đường dẫn đến thư mục chứa ảnh
image_folder = "E:/Xulyanh/Dataset" # Thay bằng đường dẫn thư mục ảnh của bạn
data, labels = prepare_data(image_folder)

# Kiểm tra nếu không có ảnh nào được xử lý thành công
if len(data) == 0 or len(labels) == 0:
    print("No valid images found. Please check your dataset path and structure.")
else:
    # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

    # Hàm đánh giá và huấn luyện từng mô hình
    def evaluate_model(model, X_train, y_train, X_test, y_test):
        start = time()
        model.fit(X_train, y_train)
        train_time = time() - start
        
        start = time()
        y_pred = model.predict(X_test)
        pred_time = time() - start
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "train_time": train_time,
            "prediction_time": pred_time
        }

    # Khởi tạo các mô hình
    svm_model = svm.SVC()
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=5)
    dt_model = tree.DecisionTreeClassifier()

    # Đánh giá mô hình SVM
    svm_results = evaluate_model(svm_model, X_train, y_train, X_test, y_test)
    print("SVM Results:", svm_results)

    # Đánh giá mô hình KNN
    knn_results = evaluate_model(knn_model, X_train, y_train, X_test, y_test)
    print("KNN Results:", knn_results)

    # Đánh giá mô hình Decision Tree
    dt_results = evaluate_model(dt_model, X_train, y_train, X_test, y_test)
    print("Decision Tree Results:", dt_results)
