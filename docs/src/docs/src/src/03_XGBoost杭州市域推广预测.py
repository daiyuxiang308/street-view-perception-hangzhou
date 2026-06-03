# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------- 路径 --------------------
train_file = r"D:\5.1test GVI\全部街景主观打分\3-四区主客观合并-有建筑15指标3000噪声删1000条-无来源标注.csv"
predict_file = r"D:\5.1test GVI\全部街景主观打分\杭州市域推广预测\XGBoost_全市预测-2\杭州市街景15指标.xlsx"

# 修改后的输出位置
output_dir = r"D:\5.1test GVI\全部街景主观打分\杭州市域推广预测\XGBoost_全市预测-2"

os.makedirs(output_dir, exist_ok=True)

# -------------------- 中文字体设置 --------------------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------- 特征与目标 --------------------
feature_cols = [
    'N01_绿视率', 'N02_水体', 'N03_天空', 'N04_自然地表',
    'B01_建筑与墙体', 'B02_障碍与桥隧',
    'T01_机动车空间', 'T02_机动车数量', 'T03_静态慢行设施', 'T04_动态慢行参与者', 'T05_交通控制设施',
    'U01_休憩与自行车设施', 'U02_安全与照明设施', 'U03_市政与便民设施', 'U04_界面信息杂乱度'
]

target_cols = ['安全', '活力', '美丽', '富裕', '无聊', '压抑']

# -------------------- 读取数据 --------------------
try:
    train_df = pd.read_csv(train_file, encoding='utf-8-sig')
except UnicodeDecodeError:
    train_df = pd.read_csv(train_file, encoding='gbk')

try:
    pred_df = pd.read_excel(predict_file)
except Exception as e:
    raise ValueError("预测文件读取失败: " + str(e))

# 去除字段名前后空格，避免字段匹配失败
train_df.columns = train_df.columns.astype(str).str.strip()
pred_df.columns = pred_df.columns.astype(str).str.strip()

# -------------------- 检查字段 --------------------
missing_train_features = [col for col in feature_cols if col not in train_df.columns]
missing_train_targets = [col for col in target_cols if col not in train_df.columns]
missing_pred_features = [col for col in feature_cols if col not in pred_df.columns]

if missing_train_features:
    raise ValueError(f"训练文件缺少特征字段：{missing_train_features}")

if missing_train_targets:
    raise ValueError(f"训练文件缺少目标字段：{missing_train_targets}")

if missing_pred_features:
    raise ValueError(f"预测文件缺少特征字段：{missing_pred_features}")

X_train = train_df[feature_cols].copy()
y_train = train_df[target_cols].copy()
X_pred = pred_df[feature_cols].copy()

# 转换为数值型
for col in feature_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_pred[col] = pd.to_numeric(X_pred[col], errors="coerce")

for col in target_cols:
    y_train[col] = pd.to_numeric(y_train[col], errors="coerce")

# 删除训练集中存在缺失的样本
train_valid = pd.concat([X_train, y_train], axis=1).dropna(subset=feature_cols + target_cols)
X_train = train_valid[feature_cols]
y_train = train_valid[target_cols]

# 预测数据如果有缺失，用训练集均值填补
X_pred = X_pred.fillna(X_train.mean())

print("=" * 80)
print("数据读取完成")
print("训练集样本数：", len(X_train))
print("预测集样本数：", len(X_pred))
print("输出路径：", output_dir)
print("=" * 80)

# -------------------- 模型参数：按照图片最终参数修改 --------------------
xgb_params = {
    'n_estimators': 750,
    'max_depth': 4,
    'learning_rate': 0.025,
    'subsample': 0.85,
    'colsample_bytree': 0.8,
    'min_child_weight': 8,
    'gamma': 0,
    'reg_alpha': 0.3,
    'reg_lambda': 8,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

# -------------------- 工具函数：保存单因子 SHAP 图 --------------------
def save_single_factor_shap(shap_values, X, target, model_name, model_output_dir):
    target_dir = os.path.join(model_output_dir, f"{target}_单因子图")
    os.makedirs(target_dir, exist_ok=True)

    for i, feature in enumerate(feature_cols):
        plt.figure(figsize=(6, 4))
        plt.scatter(
            X[feature],
            shap_values[:, i],
            alpha=0.6,
            s=15
        )
        plt.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        plt.xlabel(feature)
        plt.ylabel("SHAP value")
        plt.title(f"{model_name} {target} - {feature}")
        plt.tight_layout()

        safe_feature_name = (
            feature.replace("/", "_")
                   .replace("\\", "_")
                   .replace(":", "_")
                   .replace("*", "_")
                   .replace("?", "_")
                   .replace('"', "_")
                   .replace("<", "_")
                   .replace(">", "_")
                   .replace("|", "_")
        )

        plt.savefig(
            os.path.join(target_dir, f"{safe_feature_name}_SHAP.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


# -------------------- 对每个目标变量训练模型并预测 --------------------
train_preds = pd.DataFrame(index=X_train.index)
pred_preds = pd.DataFrame(index=pred_df.index)
feature_importance = pd.DataFrame(index=feature_cols)
metrics_list = []

for target in target_cols:
    print(f"训练并预测 {target} ...")

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train[target])

    # 训练集预测
    y_train_pred = model.predict(X_train)
    train_preds[f'pred_{target}'] = y_train_pred

    # 全市预测
    y_pred = model.predict(X_pred)

    # 如果你的主观感知分值是 1-10 分制，可以限制预测值范围
    y_pred = np.clip(y_pred, 1, 10)

    pred_preds[f'pred_{target}'] = y_pred

    # 特征重要性
    feature_importance[target] = model.feature_importances_

    # 模型指标
    metrics = {
        'Target': target,
        'Train_R2': r2_score(y_train[target], y_train_pred),
        'Train_RMSE': np.sqrt(mean_squared_error(y_train[target], y_train_pred)),
        'Train_MAE': mean_absolute_error(y_train[target], y_train_pred),
        'n_estimators': xgb_params['n_estimators'],
        'max_depth': xgb_params['max_depth'],
        'learning_rate': xgb_params['learning_rate'],
        'subsample': xgb_params['subsample'],
        'colsample_bytree': xgb_params['colsample_bytree'],
        'min_child_weight': xgb_params['min_child_weight'],
        'gamma': xgb_params['gamma'],
        'reg_alpha': xgb_params['reg_alpha'],
        'reg_lambda': xgb_params['reg_lambda']
    }

    metrics_list.append(metrics)

    # -------------------- SHAP 分析 --------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # summary plot 蜂群图
    shap.summary_plot(
        shap_values,
        X_train,
        max_display=len(feature_cols),
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{target}_SHAP_summary.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # bar plot 条形图
    shap.summary_plot(
        shap_values,
        X_train,
        plot_type="bar",
        max_display=len(feature_cols),
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{target}_SHAP_bar.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 单因子图
    save_single_factor_shap(
        shap_values=shap_values,
        X=X_train,
        target=target,
        model_name="XGBoost",
        model_output_dir=output_dir
    )

# -------------------- 保存全市预测结果 --------------------
pred_df_out = pred_df.copy()

for col in pred_preds.columns:
    pred_df_out[col] = pred_preds[col].values

pred_df_out.to_excel(
    os.path.join(output_dir, "杭州市街景15指标+打分预测.xlsx"),
    index=False
)

# -------------------- 保存训练集预测结果 --------------------
train_df_out = train_df.loc[X_train.index].copy()

for col in train_preds.columns:
    train_df_out[col] = train_preds[col].values

train_df_out.to_csv(
    os.path.join(output_dir, "训练集_带预测.csv"),
    index=False,
    encoding='utf-8-sig'
)

# -------------------- 保存特征重要性 --------------------
feature_importance.to_csv(
    os.path.join(output_dir, "特征重要性.csv"),
    encoding='utf-8-sig'
)

# -------------------- 保存模型评估指标 --------------------
pd.DataFrame(metrics_list).to_csv(
    os.path.join(output_dir, "模型评估指标.csv"),
    index=False,
    encoding='utf-8-sig'
)

# -------------------- 特征重要性热力图 --------------------
plt.figure(figsize=(12, 6))

sns.heatmap(
    feature_importance.T,
    annot=True,
    fmt=".3f",
    cmap='YlOrRd',
    cbar_kws={'label': 'Importance'}
)

plt.title("XGBoost特征重要性热力图")
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "feature_importance_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 80)
print("全市预测完成，图片与结果均已保存。")
print("输出路径：", output_dir)
print("=" * 80)
