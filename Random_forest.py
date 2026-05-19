from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from pprint import pprint

data = load_breast_cancer()
x,y = data.data, data.target

def rd(x,y):

    model = RandomForestClassifier(n_jobs=-1)
    print("bo tham so mac dinh cua mo hinh: ")
    pprint(model.get_params())

    # danh gia bang cv = 5
    loss = -cross_val_score(
        model, x, y, cv=5, scoring='neg_log_loss'
    ).mean()
    acc = cross_val_score(
        model, x, y, cv=5, scoring='accuracy'
    ).mean()

    print(f'\nLoss : {loss:.6f}')
    print(f'Accuracy    : {acc:.4f}')


# {'bootstrap': True,
#  'ccp_alpha': 0.0,
#  'class_weight': None,
#  'criterion': 'gini',
#  'max_depth': None,
#  'max_features': 'sqrt',
#  'max_leaf_nodes': None,
#  'max_samples': None,
#  'min_impurity_decrease': 0.0,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'min_weight_fraction_leaf': 0.0,
#  'monotonic_cst': None,
#  'n_estimators': 100,
#  'n_jobs': None,
#  'oob_score': False,
#  'random_state': None,
#  'verbose': 0,
#  'warm_start': False}

def PSO_rd(x,y):


    def fitness(params):
        n_estimators = int(round(params[0]))   # 50 -500
        max_depth = int(round(params[1]))       # 3- 30
        min_samples_split = int(round(params[2]))   # 2- 20
        ccp_alpha = 10** params[3]                  # -5, -1
        model = RandomForestClassifier(n_estimators= n_estimators,
                                       max_depth = max_depth,
                                       min_samples_split = min_samples_split,
                                       ccp_alpha=ccp_alpha,
                                       n_jobs=-1)
        loss = -cross_val_score(model, x, y, cv=5, scoring='neg_log_loss').mean()
        return loss

    n_particles = 10
    n_iter = 50
    dim = 4
    bounds = [(50, 500), (3, 30),(2,20), (-5,-1)]


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
        print(f'vong lap:{t + 1} | gbest_score : {gbest_score}')

if __name__ == '__main__':
    # rd(x,y)
    PSO_rd(x,y)