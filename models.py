from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Dense,
    Dropout,
    Input,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import sys
from attention_layer import Attention
from SSA import SSA, evaluate

sequence_length = 10
epochs = 30
batch_size = 32


def checkdir():
    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")
    if not os.path.exists("./images"):
        os.makedirs("./images")


def Build_LSTM(input_shape):
    print("Building LSTM model...")

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def Build_GRU(input_shape):
    print("Building GRU model...")

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(GRU(64))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def Build_BP(input_shape):
    print("Building BP model...")

    model = Sequential()
    model.add(Input(shape=(input_shape[0] * input_shape[1],)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def Build_Attention(input_shape):
    print("Building Attention model...")

    input_layer = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(input_layer)
    x = Dropout(0.2)(x)

    # Attention层
    context_vector, att_weights = Attention()(x)
    x = Dense(64, activation="relu")(context_vector)
    x = Dropout(0.2)(x)
    output_layer = Dense(1)(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def Build_SSA(part, X_train, y_train, X_test, y_test):
    print("Building SSA model...")

    # 超参数搜索空间
    n_dim = 3
    lb = [0.0008, 1, 16]  # 学习率下限，层数下限，神经元下限
    ub = [0.2, 4, 256]  # 学习率上限，层数上限，神经元上限

    # SSA参数
    pop_size = 15
    max_iter = 25
    verbose = True

    ssa = SSA(
        func=lambda x: evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            learning_rate=x[0],
            num_layers=int(round(x[1])),
            uints=int(round(x[2])),
        ),
        n_dim=n_dim,
        pop_size=pop_size,
        max_iter=max_iter,
        lb=lb,
        ub=ub,
        verbose=verbose,
    )
    best_x, best_y = ssa.run()

    checkdir()
    output_file = f"./outputs/best_param{part + 1}.txt"
    with open(output_file, "a") as f:
        sys.stdout = f

        print(f"alpha: {best_x[0]:.6f}")
        print(f"n_layers: {int(round(best_x[1]))}")
        print(f"uints: {int(round(best_x[2]))}")
        print(f"minimum val_loss: {best_y:.4f}")

        sys.stdout = sys.__stdout__

    plt.rcParams["font.sans-serif"] = ["SimHei"]  # For Chinese characters
    plt.rcParams["axes.unicode_minus"] = False  # For proper minus sign display
    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(ssa.gbest_y_hist, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Loss")
    plt.grid(True)

    checkdir()
    plt.savefig(f"./images/vergence_curve{part}.png")
    plt.close()

    learning_rate = best_x[0]
    n_layers = int(best_x[1])
    units = int(best_x[2])

    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = LSTM(units, return_sequences=True)(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    for _ in range(1, n_layers - 1):
        x = LSTM(units, return_sequences=True)(x)
        x = Dropout(0.2)(x)

    if n_layers > 1:
        x = LSTM(units, return_sequences=True)(x)
        x = Dropout(0.2)(x)

    context_vector, att_weights = Attention()(x)
    x = Dense(64, activation="relu")(context_vector)
    x = Dropout(0.2)(x)
    output_layer = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")

    return model
