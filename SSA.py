from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from attention_layer import Attention


"""
Ref:https://github.com/changliang5811/SSA_python
Ref:https://www.tandfonline.com/doi/full/10.1080/21642583.2019.1708830
Ref:A novel swarm intelligence optimization approach: sparrow search algorithm.pdf
"""


def evaluate(X_train, y_train, X_test, y_test, learning_rate, num_layers, uints):
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(64, return_sequences=True)(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    for _ in range(num_layers - 1):
        x = LSTM(int(round(uints)), return_sequences=True)(x)
        x = Dropout(0.2)(x)

    if num_layers > 1:
        x = LSTM(int(round(uints)), return_sequences=True)(x)
        x = Dropout(0.2)(x)

    context_vector, att_weights = Attention()(x)
    x = Dense(64, activation="relu")(context_vector)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error"
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=0,
    )
    val_rmse = np.sqrt(history.history["val_loss"])
    return np.mean(val_rmse)


class SSA:
    def __init__(
        self,
        func,
        n_dim=None,
        pop_size=15,
        max_iter=25,
        lb=None,
        ub=None,
        verbose=False,
    ):
        self.func = func
        self.n_dim = n_dim
        self.pop = pop_size
        P_percent = 0.2
        D_percent = 0.1
        self.pNum = round(self.pop * P_percent)
        self.warn = round(self.pop * D_percent)

        self.max_iter = max_iter
        self.verbose = verbose

        self.lb, self.ub = np.array(lb), np.array(ub)
        assert self.n_dim == len(self.lb) == len(self.ub), "参数维度错误"
        assert np.all(self.ub > self.lb), "上限必须大于下限"

        self.X = np.random.uniform(
            low=self.lb, high=self.ub, size=(self.pop, self.n_dim)
        )
        self.Y = [self.func(self.X[i]) for i in range(self.pop)]

        self.pbest_x = self.X.copy()
        self.pbest_y = np.full(self.pop, np.inf)
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)
        self.gbest_y = np.inf
        self.gbest_y_hist = []
        self.update_pbest()
        self.update_gbest()

        self.idx_max = 0
        self.x_max = self.X[self.idx_max, :]
        self.y_max = self.Y[self.idx_max]

    def cal_y(self, start, end):
        for i in range(start, end):
            self.Y[i] = self.func(self.X[i])

    def update_pbest(self):
        improved = np.array(self.Y) < self.pbest_y
        self.pbest_x[improved] = self.X[improved]
        self.pbest_y[improved] = np.array(self.Y)[improved]

    def update_gbest(self):
        idx_min = np.argmin(self.pbest_y)
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]

    def find_worst(self):
        self.idx_max = np.argmax(self.Y)
        self.x_max = self.X[self.idx_max, :]
        self.y_max = self.Y[self.idx_max]

    def update_finder(self, iter_num):
        sorted_indices = np.argsort(self.Y)
        for i in range(self.pNum):
            idx = sorted_indices[i]
            r2 = np.random.rand()
            if r2 < 0.8:
                r1 = np.random.rand()
                self.X[idx] *= np.exp(-iter_num / (r1 * self.max_iter))
            else:
                self.X[idx] += np.random.normal(0, 1, self.n_dim)
            self.X[idx] = np.clip(self.X[idx], self.lb, self.ub)
        self.cal_y(0, self.pNum)

    def update_follower(self):
        sorted_indices = np.argsort(self.Y)
        best_idx = sorted_indices[0]
        bestXX = self.X[best_idx]

        for i in range(self.pNum, self.pop):
            idx = sorted_indices[i]
            if i > self.pop / 2:
                self.X[idx] = np.random.rand() * np.exp(
                    (self.x_max - self.X[idx]) / i**2
                )
            else:
                A = np.random.choice([-1, 1], size=self.n_dim)
                self.X[idx] = (
                    bestXX + np.abs(self.X[idx] - bestXX).dot(A) / (A.dot(A)) * A
                )
            self.X[idx] = np.clip(self.X[idx], self.lb, self.ub)
        self.cal_y(self.pNum, self.pop)

    def detect(self):
        candidates = np.random.choice(self.pop, self.warn, replace=False)
        for j in candidates:
            if self.Y[j] > self.gbest_y:
                self.X[j] = self.gbest_x + np.random.normal(0, 1, self.n_dim) * np.abs(
                    self.X[j] - self.gbest_x
                )
            else:
                self.X[j] += (
                    (2 * np.random.rand() - 1)
                    * np.abs(self.X[j] - self.x_max)
                    / (self.Y[j] - self.y_max + 1e-10)
                )
            self.X[j] = np.clip(self.X[j], self.lb, self.ub)
            self.Y[j] = self.func(self.X[j])

    def run(self, tolerance=1e-8, patience=2):
        counter = 0

        for iter_num in range(self.max_iter):
            self.update_finder(iter_num)
            self.find_worst()
            self.update_follower()
            self.update_pbest()
            self.update_gbest()
            self.detect()
            self.update_pbest()
            self.update_gbest()
            self.gbest_y_hist.append(self.gbest_y)
            if self.verbose:
                print(f"Iter {iter_num}, Best Loss: {self.gbest_y}")

            if len(self.gbest_y_hist) > 1:
                if abs(self.gbest_y_hist[-1] - self.gbest_y_hist[-2]) < tolerance:
                    counter += 1
                else:
                    counter = 0

            if counter >= patience:
                print(f"Early stopping triggered at iteration {iter_num + 1}.")
                break
        return self.gbest_x, self.gbest_y
