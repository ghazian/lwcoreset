import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

class GeometricDecomposition:
    def __init__(self, X, n, k, eps):
        self.X = X
        self.n = n
        self.k = k
        self.eps = eps

    def set_k(self, k):
        self.k = k

    def set_X(self, X):
        self.X = X

    def compute_ff_coreset(self, X):
        # Compute fast factor approximation coreset
        n = X.shape[0]
        k = int(2 * np.ceil(np.power(n, 1 / 4)))

        A = self.farthest_point_alg(k)
        L = int(np.amax(np.sqrt(np.power((np.ndarray.copy(X).sum(axis=1))))))

        # Picking random sample from X
        gamma = 3
        sample_size = int(np.ceil(gamma * k * np.log(n)))
        if sample_size >= n:
            return X
        index = np.random.choice(X.shape[0], sample_size, replace=False)
        B = X[index]

        # Approx centers set
        C = np.unique(np.concatenate((A, B), axis=0), axis=0)
        return C, L

    def farthest_point_alg(self, k):
        #Gonzalez algorithm
        X = np.ndarray(self.X)
        index = np.random.choice(X.shape[0], 1, replace=False)
        T = np.array(X[index])
        X = np.delete(X, index, axis=0)
        distance = np.ndarray(shape=X.shape[0])
        distance.fill(np.NINF)

        while T.size < k*2:
            distance_new = np.sqrt(np.power(X-T[-1], 2).sum(axis=1))
            distance = np.maximum(distance, distance_new)
            D = np.amax(distance)
            index = np.where(distance == D)
            T = np.append(T, X[index[0]], axis=0)
            X = np.delete(X, index[0], axis=0)
            distance = np.delete(distance, index[0], axis=0)

        return T

    def calcNN(self, X, C):
        neighbors= NearestNeighbors(n_neighbors=2, algorithm='auto').fit(C)
        distance, indices = neighbors.kneighbors(X)
        return distance, indices

    def calcNC(self, X, C):
        P = np.ndarray(shape=(X.shape))
        for index, point in enumerate(X):
            distance = np.inf
            center = [0,0]
            for c in C:
                new_dist = np.power(point-center, 2).sum()
                if new_dist < distance:
                    center = c
                    dist = new_dist
            P[index] = center
        return P

    def exp_grid(self, p, P, R):
        # Unweighted version
        af = 32
        D = np.sqrt(np.power(P-p, 2).sum(axis=1))
        S = np.array([[0,0]])
        W = np.array([0])

        index = np.where(D < R)
        P_temp = P[index[0]]

        if(len(P_temp) > 0):
            idx = np.random.choice(len(P_temp), 1, replace=False)
            S = np.append(S, P_temp(idx), axis=0)
            W = np.append(W, [len(P_temp)], axis=0)

        R_temp = R
        limit = int(2*np.log(af*self.n))
        for i in range(limit):
            R_temp *= 2
            index = np.where((R_temp / 2 < D) & (D < R_temp))

            ### diagram ###
            fig = plt.Circle((p[0], p[1]), R_temp, fill=False)
            ax = plt.gca()
            ax.add_artist(fig)

            if len(index[0]) > 0:
                candidates = P[index[0]]
                size_temp = int(np.ceil(np.log(len(index[0]))))
                idx = np.random.choice(len(candidates), size_temp, replace=False)
                S = np.append(S, candidates[idx], axis=0)
        if P.shape[0] > 10:
            plt.scatter(S[1:, 0], S[1:, 1])
            plt.show()
        return S[1:], W[1:]

    def good_set(self, X, C, L, info):
        n = X.shape[0]
        index = np.random.choice(C.shape[0], 1, replace=False)
        P = np.array(C[index])
        ub = int(L / (4 * n))
        while True:
            indicates = info[:, 0] < ub
            ub *= 2

            P_i = X[indicates]
            P = np.unique(np.concatenate((P, P_i), axis=0), axis=0)
            if (P.shape[0] >= n / 2):
                break

        for point in P:
            index = np.where(X == point)
            if len(index[0]):
                X = np.delete(X, index[0][0], axis=0)
        return X

    def bad_points(self, X, C, L):
        #Further development of the code needed
        distance, index = self.calcNN(X, C)

    def c_map(self, dist, index, P):
        info = np.ndarray(shape=(dist.shape[0]), dtype=int)
        point = np.ndarray(shape=(dist.shape))

        for i in range(dist.shape[0]):
            if dist[i][0] > dist[i][1]:
                info[i] = int(dist[i][0])
                point[i] = P[index[i][0]]
            else:
                info[i] = int(dist[i][1])
                point[i] = P[index[i][1]]

        return info, point

    def centroid_set(self):
        #Har-Peled & Mazumdar algorithm for computing centroid set
        approx_factor = 32

        # Computing fast constant factor approximation algorithm
        X = np.ndarray.copy(self.X)
        index = np.random.choice(X.shape[0], 1, replace=False)
        A = np.array(X[index])
        n = X.shape[0]
        iter = np.floor(np.log(n))
        X = np.delete(X, index, axis=0)
        min_size = int(n/(np.log(n)))

        while iter:
            C,L = self.compute_ff_coreset(X)
            A = np.unique(np.concatenate((A, C), axis=0), axis=0)
            X = self.bad_points(X,C,L)
            iter -= 1
            if X.shape[0] < min_size:
                A = np.unique(np.concatenate((A,X), axis=0), axis=0)
                break

        X = np.ndarray.copy(self.X)
        for p in A:
            index = np.where(X == p)
            X = np.delete(X, index[0], axis=0)

        # Computing the final coreset construction from approximating clustering
        dist,index = self.calcNN(X,A)
        point = self.c_map(dist, index, A)

        value = np.power(X-point, 2).sum(axis=1).sum()
        Rd = int(np.sqrt(value/(approx_factor*self.n)))

        S = np.array([[0,0]])
        W = np.array([0])

        for p in A:
            index = np.where(point == p)
            if (len(index[0]) > 0):
                P = np.unique(X[index[0]], axis=0)
                S_temp, W_temp = self.exp_grid(p,P,Rd)
                S = np.append(S, S_temp, axis=0)
                W = np.append(W, W_temp, axis=0)

        return S[1:], W[1:]