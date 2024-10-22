import numpy as np
from scipy.spatial.distance import cdist
class KMean_Clustering:
    def __init__(self, K = 3):
        self.K = K
        self.centroids = None
        # Lấy ra K mẫu đầu tiên trong X_Train làm tâm cụm
    def kmeans_init_centroids(self, X):
        arr = []
        for i in range (self.K):
            arr.append(i+1)
        return X[arr]
        # Tính toán khoảng cách từng mẫu dữ liệu đến từng tâm cụm, phân mẫu dữ liệu về cụm có khoảng cách nhỏ nhất
    def kmeans_assign_labels(self, X, centroids):
        D = cdist(X, centroids)
        return np.argmin(D, axis = 1)
        # Kiểm tra tâm lần lặp sau có bằng tâm lần lặp trước không???
    def has_converged(self, centroids, new_centroids):
        return (set([tuple(a) for a in centroids]) == set([tuple(a) for a in new_centroids]))
        # Tính tâm mới bằng cách lấy trung bình cộng các điểm trong cụm
    def kmeans_update_centroids(self, X, labels):
        centroids = np.zeros((self.K, X.shape[1]))
        for k in range(self.K):
            Xk = X[labels == k, :]
            centroids[k,:] = np.mean(Xk, axis = 0)
        return centroids

    def fit(self, X):
        # Khởi tạo tâm cụm
        self.centroids = [self.kmeans_init_centroids(X)]
        labels = []
        it = 0
        while True:
            # Tính toán khoảng cách các mẫu dữ liệu đến các tâm, phân về cụm có khoảng cách nhỏ nhất
            labels.append(self.kmeans_assign_labels(X, self.centroids[-1]))
            # Cập nhật lại các tâm
            new_centroids = self.kmeans_update_centroids(X, labels[-1])
            # Kiểm tra tâm mới và tâm cũ có == nhau không
            # Nếu có thì kết thúc thuật toán
            if self.has_converged(self.centroids[-1], new_centroids):
                break
            # Thêm tâm cụm mới vào mảng centroids
            self.centroids.append(new_centroids)
            it += 1
        return (self.centroids, labels, it)
    # Dự đoán điểm dữ liệu thuộc về cụm nào
    # Tính toán khoảng cách các mẫu dữ liệu đến các tâm, phân về cụm có khoảng cách nhỏ nhất
    def predict(self, X_Test):
        labels = []
        labels.append(self.kmeans_assign_labels(X_Test, self.centroids[-1]))
        return labels
