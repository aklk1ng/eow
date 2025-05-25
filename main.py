import pandas as pd
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from models import (
    checkdir,
    Build_LSTM,
    Build_GRU,
    Build_BP,
    Build_Attention,
    Build_SSA,
)

epochs = 30
batch_size = 16

data = pd.read_excel("./data.xlsx")


data["datetime"] = pd.to_datetime(data["datetime"])
data["hour"] = data["datetime"].dt.hour
data["day_of_week"] = data["datetime"].dt.dayofweek
data["month"] = data["datetime"].dt.month
holidays = [
    "2019-04-19",  # Good Friday (耶稣受难日)
    "2019-04-22",  # Easter Monday (复活节星期一)
    "2019-05-06",  # Early May Bank Holiday (五月初银行假日)
    "2019-05-27",  # Spring Bank Holiday (春季银行假日)
]
data["is_holiday"] = data["datetime"].apply(
    lambda date: 1 if str(date.date()) in holidays else 0
)

features = data[
    ["Flow", "Avg mph", "hour", "day_of_week", "month", "is_holiday"]
].values
target = data["Flow"].values.reshape(-1, 1)

# 归一化
scaler_features = MinMaxScaler(feature_range=(0, 1))
features_scaled = scaler_features.fit_transform(features)

scaler_target = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler_target.fit_transform(target)


def plot_attention_weights(traffic_attention, time_attention, save_path=None):
    """
    可视化交通特征和时间特征的注意力权重

    Args:
        traffic_attention: 交通特征的注意力权重
        time_attention: 时间特征的注意力权重
        save_path: 保存图片的路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 可视化交通特征注意力
    traffic_att = traffic_attention.numpy().squeeze()
    sns.heatmap(
        traffic_att.T,
        ax=ax1,
        cmap="YlOrRd",
        xticklabels=10,  # 每10个时间步显示一个标签
        yticklabels=["Flow", "Avg mph"],
    )
    ax1.set_title("交通特征注意力分布")
    ax1.set_xlabel("时间步")
    ax1.set_ylabel("特征")

    # 可视化时间特征注意力
    time_att = time_attention.numpy().squeeze()
    sns.heatmap(
        time_att.T,
        ax=ax2,
        cmap="YlOrRd",
        xticklabels=10,
        yticklabels=["Hour", "Day of Week", "Month", "Is Holiday"],
    )
    ax2.set_title("时间特征注意力分布")
    ax2.set_xlabel("时间步")
    ax2.set_ylabel("特征")

    plt.tight_layout()

    checkdir()
    plt.savefig(save_path)
    plt.close()


def create_sequences(features, target, sequence_length):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i : i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)


sequence_length = 10
X, y = create_sequences(features_scaled, target_scaled, sequence_length)


# 数据分割
parts = 2
split_index = int(len(X) * 1 / parts)
train_size = int(split_index * 0.8)

X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []

for i in range(parts):
    start_index = i * split_index
    end_index = (i + 1) * split_index

    X_train = X[start_index:end_index][:train_size]
    X_test = X[start_index:end_index][train_size:]
    y_train = y[start_index:end_index][:train_size]
    y_test = y[start_index:end_index][train_size:]

    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)


model_results = []
for i in range(parts):
    print(f"Part {i + 1}:")
    X_train, y_train, X_test, y_test = (
        X_train_list[i],
        y_train_list[i],
        X_test_list[i],
        y_test_list[i],
    )
    y_test_rescaled = scaler_target.inverse_transform(y_test).flatten()
    input_shape = (sequence_length, X_train.shape[2])

    # 构建模型
    lstm_model = Build_LSTM(input_shape)
    gru_model = Build_GRU(input_shape)
    bp_model = Build_BP(input_shape)
    attention_model = Build_Attention(input_shape)
    ssa_model = Build_SSA(i, X_train, y_train, X_test, y_test)

    # 训练模型
    print(f"Training LSTM for part {i + 1}")
    lstm_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )
    print(f"Training GRU for part {i + 1}")
    gru_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    print(f"Training BP for part {i + 1}")
    bp_model.fit(
        X_train_flat,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_flat, y_test),
    )
    print(f"Training Attention for part {i + 1}")
    attention_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )
    print(f"Training SSA for part {i + 1}")
    ssa_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )

    # 预测
    lstm_predictions = lstm_model.predict(X_test).flatten()
    gru_predictions = gru_model.predict(X_test).flatten()
    bp_predictions = bp_model.predict(X_test_flat).flatten()
    attention_predictions = attention_model.predict(X_test).flatten()
    ssa_predictions = ssa_model.predict(X_test).flatten()

    # plot_attention_weights(
    #     traffic_att, time_att, f"./images/attention_weights_part{i + 1}.png"
    # )

    # 反归一化
    lstm_predictions_rescaled = scaler_target.inverse_transform(
        lstm_predictions.reshape(-1, 1)
    ).flatten()
    gru_predictions_rescaled = scaler_target.inverse_transform(
        gru_predictions.reshape(-1, 1)
    ).flatten()
    bp_predictions_rescaled = scaler_target.inverse_transform(
        bp_predictions.reshape(-1, 1)
    ).flatten()
    attention_predictions_rescaled = scaler_target.inverse_transform(
        attention_predictions.reshape(-1, 1)
    ).flatten()
    ssa_predictions_rescaled = scaler_target.inverse_transform(
        ssa_predictions.reshape(-1, 1)
    ).flatten()

    # 计算 RMSE 和 MAPE
    lstm_rmse = np.sqrt(mean_squared_error(y_test_rescaled, lstm_predictions_rescaled))
    lstm_mape = (
        np.mean(np.abs((y_test_rescaled - lstm_predictions_rescaled) / y_test_rescaled))
        * 100
    )
    gru_rmse = np.sqrt(mean_squared_error(y_test_rescaled, gru_predictions_rescaled))
    gru_mape = (
        np.mean(np.abs((y_test_rescaled - gru_predictions_rescaled) / y_test_rescaled))
        * 100
    )
    bp_rmse = np.sqrt(mean_squared_error(y_test_rescaled, bp_predictions_rescaled))
    bp_mape = (
        np.mean(np.abs((y_test_rescaled - bp_predictions_rescaled) / y_test_rescaled))
        * 100
    )
    attention_rmse = np.sqrt(
        mean_squared_error(y_test_rescaled, attention_predictions_rescaled)
    )
    attention_mape = (
        np.mean(
            np.abs((y_test_rescaled - attention_predictions_rescaled) / y_test_rescaled)
        )
        * 100
    )
    ssa_rmse = np.sqrt(mean_squared_error(y_test_rescaled, ssa_predictions_rescaled))
    ssa_mape = (
        np.mean(np.abs((y_test_rescaled - ssa_predictions_rescaled) / y_test_rescaled))
        * 100
    )

    model_results.append(
        {
            "part": i + 1,
            "LSTM": {
                "predictions": lstm_predictions_rescaled,
                "rmse": lstm_rmse,
                "mape": lstm_mape,
            },
            "GRU": {
                "predictions": gru_predictions_rescaled,
                "rmse": gru_rmse,
                "mape": gru_mape,
            },
            "BP": {
                "predictions": bp_predictions_rescaled,
                "rmse": bp_rmse,
                "mape": bp_mape,
            },
            "LSTM+Attention": {
                "predictions": attention_predictions_rescaled,
                "rmse": attention_rmse,
                "mape": attention_mape,
            },
            "LSTM+Attention+SSA": {
                "predictions": ssa_predictions_rescaled,
                "rmse": ssa_rmse,
                "mape": ssa_mape,
            },
        }
    )

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
for results in model_results:
    part = results["part"]
    y_test_rescaled = scaler_target.inverse_transform(y_test_list[part - 1]).flatten()

    # 绘制预测结果
    plt.figure(figsize=(12, 8))
    plt.plot(y_test_rescaled[:100], label="Actual Values", color="blue", linewidth=2)
    plt.plot(
        results["LSTM"]["predictions"][:100],
        label="LSTM",
        color="red",
        linestyle="-",
        linewidth=2,
    )
    plt.plot(
        results["GRU"]["predictions"][:100],
        label="GRU",
        color="purple",
        linestyle="-",
        linewidth=2,
    )
    plt.plot(
        results["BP"]["predictions"][:100],
        label="BP",
        color="black",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        results["LSTM+Attention"]["predictions"][:100],
        label="LSTM+Attention",
        color="green",
        linestyle="-",
        linewidth=2,
    )
    plt.plot(
        results["LSTM+Attention+SSA"]["predictions"][:100],
        label="LSTM+Attention+SSA",
        color="orange",
        linestyle="-",
        linewidth=2,
    )
    plt.xlabel("采样时间/(15)min", fontsize=14)
    plt.ylabel("车流量", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    checkdir()
    plt.savefig(f"./images/traffic_flow_prediction{part}.png")
    plt.close()

    checkdir()
    output_file = f"./outputs/output{part}.txt"
    with open(output_file, "a") as f:
        sys.stdout = f

        print(f"Part {part} Results:")
        print(f"{'Model':<20}{'RMSE':<15}{'MAPE (%)':<15}")
        print("-" * 50)
        for model_name in ["LSTM", "GRU", "BP", "LSTM+Attention", "LSTM+Attention+SSA"]:
            rmse = results[model_name]["rmse"]
            mape = results[model_name]["mape"]
            print(f"{model_name:<20}{rmse:<15.2f}{mape:<15.2f}")
        print("\n")

        sys.stdout = sys.__stdout__
