import numpy as np

def euclidean_dist(v1, v2):
    return np.sqrt(np.sum((v2-v1)**2, axis=0))

def manhattan_dist(v1, v2):
    return np.sum(abs(v2-v1), axis=0)

class DBScan():
    """
    Density-Based Spatial Clustering Algorithm
    Shluk je definován jako maximální množina hustě spojených bodů.
    """
    soliter_value = -1

    def __init__(self, eps: int = 0.5, min_samples: int = 4):
        self.eps = eps
        self.min_samples = min_samples
        self.clusters = None

    def _get_neighbours(self, point) -> np.ndarray:
        """
        Vrátí všechny indexy bodů v okolí bodu point.
        """
        return np.where(np.linalg.norm(self.data - point, axis=1) < self.eps)[0]

    def _make_cluster(self, index):
        """
        Add all conected points to cluster.
        """
        self.cluster_id += 1
        self.labels_[index] = self.cluster_id
        neighbours_id = list(self._get_neighbours(self.data[index]))
        while len(neighbours_id) > 0:
            neighbour_id = neighbours_id.pop()
            self.labels_[neighbour_id] = self.cluster_id
            for new_id in list(self._get_neighbours(self.data[neighbour_id])):
                # Add index to neighbours
                if self.labels_[new_id] == 0 and new_id not in neighbours_id:
                    neighbours_id.append(new_id)
                #print(neighbours_id)

    def fit(self, data) -> None:
        """
        ## Algoritmus:
        1. Vybrat počáteční bod p
        2. Získat všechny nepřímo dosažitelné body
            a) Pokud je bod p core bod -> cluster hotový
            b) Pokud je p border bod, tak z něj ostatní body nemohou být nepřímo
               dosažitelné -> vybrat další počáteční bod z databáze
        3. Pokračujeme, dokud jsme neprošli všechny body
        """
        self.data = data
        self.labels_ = np.zeros(self.data.shape[0])
        self.cluster_id = 0

        for index, point in enumerate(self.data):
            neighbours_id = self._get_neighbours(point)
            if len(neighbours_id) == 0:    # soliter
                self.labels_[index] = DBScan.soliter_value
            if self.labels_[index] != 0:
                continue
            elif len(neighbours_id) >= self.min_samples:
                self._make_cluster(index)
            #print(self.labels_)
        
# END
