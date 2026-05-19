import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from pprint import pprint
import matplotlib.pyplot as plt

#load du lieu
data = load_breast_cancer()
x, y = data.data, data.target

def logistic_regression(x, y):
    scale    = StandardScaler()
    x_scaled = scale.fit_transform(x)

    model = LogisticRegression(max_iter=5000)
    print('bo tham so mac dinh: ')
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

def PSO_logistic_regression(x,y):
    # chuan hoa du lieu
    scale = StandardScaler()
    x_train = scale.fit_transform(x)

    # solver_map = {
    #     0: 'liblinear',
    #     1: 'saga'
    # }
    # penalty_map = {
    #     0: 'l1',
    #     1: 'l2'
    # }
    solver_map = {
        0: 'liblinear',
        1: 'lbfgs',  # ← thêm solver tốt nhất
        2: 'saga'
    }
    penalty_map = {
        0: 'l2',  # l2 dùng được với cả 3 solver
        1: 'l1'  # l1 chỉ dùng được với liblinear và saga
    }
    # xay dung ham muc tieu
    # def fitness(params):
    #     C = 10 ** params[0]  # C thuoc  10^(-3,2)
    #     solver = solver_map[int(round(params[1]))]  # thuoc 0 ->1
    #     penalty = penalty_map[int(round(params[2]))]  # thuoc 0 -> 2
    #
    #     model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=5000)
    #     neg_loss = cross_val_score(model, x_train, y, cv=5, scoring='neg_log_loss').mean()
    #     loss = -neg_loss
    #     return loss

    def fitness(params):
        C = 10 ** params[0]
        solver = solver_map[int(round(np.clip(params[1], 0, 2)))]
        penalty = penalty_map[int(round(np.clip(params[2], 0, 1)))]

        # lbfgs không hỗ trợ l1 → phạt nặng
        if solver == 'lbfgs' and penalty == 'l1':
            return 999.0

        try:
            model = LogisticRegression(C=C, penalty=penalty,
                                       solver=solver, max_iter=5000)
            neg_loss = cross_val_score(model, x_train, y,
                                       cv=5, scoring='neg_log_loss').mean()
            return -neg_loss
        except Exception:
            return 999.0

    # khoi tao
    n_particles =  20                # so ca the
    n_iter = 50                       # so vong lap
    dim = 3                           # so sieu tham so can toi uu
    # bounds = [(-3,2), (0,1),(0,1)]    # khong gian tim kiem
    bounds = [(-3, 2), (0, 2), (0, 1)]

    #khoi tao bo particle ban dau
    positions = np.random.uniform(
        [b[0] for b in bounds],
        [b[1] for b in bounds],
        (n_particles, dim)
    )
    # khoi tao van toc ban dau
    velocities = np.zeros((n_particles, dim))

    # khoi tao pbest va gbest ban dau
    pbest = positions.copy()
    pbest_score = np.array([fitness(p) for p in pbest])
    gbest = pbest[np.argmin(pbest_score)].copy()
    gbest_scores = np.min(pbest_score)

    #khoi tao cac gia tri
    w, c1, c2 = 0.7, 1.5, 1.5

    # history = []
    # khoi tao thuat toan
    for t in range(n_iter):
        r1, r2 = np.random.rand(n_particles, dim), np.random.rand(n_particles, dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest - positions)
                      + c2 * r2 * (gbest - positions))
        positions += velocities
        # Clamp ve bounds
        for d, (lo, hi) in enumerate(bounds):
            positions[:, d] = np.clip(positions[:, d], lo, hi)
        # Đánh giá
        scores = np.array([fitness(p) for p in positions])
        # print(f'vong lap:{t + 1 } | scores : {scores}')
        improved = scores < pbest_score
        pbest[improved] = positions[improved]
        pbest_score[improved] = scores[improved]
        gbest = pbest[np.argmin(pbest_score)].copy()
        gbest_score = np.min(pbest_score)
        # print(r1, r2)
        # print(pbest)
        # print(gbest)
        # print(pbest_score)
        print(f'vong lap:{t + 1 } | gbest_score : {gbest_score}')
        # history.append(gbest_scores)
        # # break
        # print(f'vong lap:{t + 1} | gbest_score: {gbest_score:.6f} '
        #       f'| solver: {solver_map[int(round(gbest[1]))]} '
        #       f'| penalty: {penalty_map[int(round(gbest[2]))]} '
        #       f'| C: {10 ** gbest[0]:.4f}')

    # print('gia tri gbest_score: ', gbest_scores)
    # print()
    # plt.plot(history)
    # plt.show()
if __name__ == '__main__':
    # logistic_regression(x,y)
    PSO_logistic_regression(x,y)

