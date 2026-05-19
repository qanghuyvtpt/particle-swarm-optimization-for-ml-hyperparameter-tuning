import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from pprint import pprint

data = load_breast_cancer()
x, y = data.data, data.target

def SVC_model(x,y):
    scale = StandardScaler()
    x_scaled = scale.fit_transform(x)
    model = SVC(probability=True)
    print('bo tham so mac dinh cua mo hinh: ')
    pprint(model.get_params())

    # danh gia bang cv = 5
    loss = -cross_val_score(
        model, x_scaled, y, cv=5, scoring='neg_log_loss'
    ).mean()
    acc = cross_val_score(
        model, x_scaled, y, cv=5, scoring='accuracy'
    ).mean()

    print(f'Loss : {loss:.6f}')
    print(f'Accuracy    : {acc:.4f}')

def SVC_model_PSO(x,y):
    scale = StandardScaler()
    x_scaled = scale.fit_transform(x)

    def fitness(params):
        C = 10 ** params[0]
        gamma = 10 ** params[1]
        model = SVC(C=C, gamma=gamma, probability=True)
        loss = -cross_val_score(model, x_scaled, y, cv=5, scoring='neg_log_loss').mean()
        return loss

    n_particles = 20
    n_iter = 50
    dim = 2  # [log10(C), log10(gamma)]
    bounds = [(-2, 3), (-4, 1)]

    positions = np.random.uniform(
        [b[0] for b in bounds], [b[1] for b in bounds], (n_particles, dim))
    velocities = np.zeros((n_particles, dim))
    pbest = positions.copy()
    pbest_scores = np.array([fitness(p) for p in pbest])
    gbest = pbest[np.argmin(pbest_scores)]

    w, c1, c2 = 0.7, 1.5, 1.5

    for t in range(n_iter):
        r1, r2 = np.random.rand(n_particles, dim), np.random.rand(n_particles, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest - positions)
                      + c2 * r2 * (gbest - positions))
        positions += velocities
        # Clamp về bounds
        for d, (lo, hi) in enumerate(bounds):
            positions[:, d] = np.clip(positions[:, d], lo, hi)
        # Đánh giá
        scores = np.array([fitness(p) for p in positions])
        improved = scores < pbest_scores
        pbest[improved] = positions[improved]
        pbest_scores[improved] = scores[improved]
        gbest = pbest[np.argmin(pbest_scores)]
        gbest_score = np.min(pbest_scores)
        print(f'vong lap:{t + 1 } | gbest_score : {gbest_score}')


if __name__ == '__main__':
    # SVC_model(x,y)
    SVC_model_PSO(x, y)