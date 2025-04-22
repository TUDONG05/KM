import time
import numpy as np
from utility import * 
class KMeans:
    def __init__(self,X,n_clusters,max_iter,m, epsilon, seed):
        self.X= np.array(X)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.seed = seed
        self.n_data, self.n_features = self.X.shape
        self.centroids = self._init_centroids()
        self._label = np.zeros(self.n_data)
        self.m =m
        self.process_time =0


    def _init_centroids(self):
        """Chọn ngẫu nhiên n_clusters điểm từ dữ liệu làm tâm cụm ban đầu"""
        np.random.seed(self.seed)
        return self.X[np.random.choice(self.n_data, self.n_clusters, replace=False)]
    
    def _update_label(self):
        d = norm_distances(self.X[:,np.newaxis], self.centroids, axis=2)
        return np.argmin(d,axis =1)

    def _update_centroids(self):
        """Cập nhật tâm cụm"""
        for i in range(self.n_clusters):
            self.centroids[i] = np.mean(self.X[self._label == i], axis=0)
        return self.centroids
    
    def _check_convergence(self):
        """Kiểm tra hội tụ"""
        d = norm_distances(self.centroids[:,np.newaxis], self.centroids, axis=2)
        return np.all(d < self.epsilon)
    def _fit(self):
        start = time.time()
        for i in range(self.max_iter):
            self._label = self._update_label()
            self.centroids = self._update_centroids()
            if self._check_convergence():
                break
        end = time.time()
        self.process_time = end - start
        self.step = i + 1
        return self.centroids, self._label

if __name__ == '__main__':
    import time
    from utility import round_float, extract_labels
    from dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
    from validity import dunn, davies_bouldin, partition_coefficient,Xie_Benie,classification_entropy, silhouette,hypervolume,accuracy_score

    ROUND_FLOAT = 3
    EPSILON = 1e-5
    MAX_ITER = 100
    M = 2
    SEED = 42
    SPLIT = '\t'
    # =======================================

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def write_report(alg: str, index: int, process_time: float, step: int, X: np.ndarray, V: np.ndarray, labels: np.ndarray) -> str:
        
        kqdg = [
            alg,
            wdvl(process_time, n=2),
            str(step),
            wdvl(dunn(X, labels)),  # DI
            wdvl(davies_bouldin(X, labels)),  # DB
            # wdvl(partition_coefficient(U)),  # PC
            # wdvl(Xie_Benie(X, V, U)),  # XB
            # wdvl(classification_entropy(U)), # CE
            wdvl(silhouette(X, labels)), #SI
            # wdvl(hypervolume(U,m=2)), # FHV
        
        ]
        return SPLIT.join(kqdg)
    # =======================================

    clustering_report = []
    data_id = 53
    if data_id in TEST_CASES:
        _start_time = time.time()
        _TEST = TEST_CASES[data_id]
        _dt = fetch_data_from_local(data_id)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        X, Y = _dt['X'], _dt['Y']
        _size = f"{_dt['data']['num_instances']} x {_dt['data']['num_features']}"
        print(f'size={_size}')
        n_clusters = _TEST['n_cluster']
        # ===============================================

        km = KMeans(X, n_clusters=n_clusters, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED,m=M)
        centroids, labels = km._fit()
        # print(f"Centroids: {centroids}")
        # print(f"Labels: {labels}")
        
        titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-','SI+']
        print(SPLIT.join(titles))
        print(write_report(alg='Kmeans', index=0, process_time=km.time, step=km.step, X=X, V=km.centroids, labels=km._label))

        