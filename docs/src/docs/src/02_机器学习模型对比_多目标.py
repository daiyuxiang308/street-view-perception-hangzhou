# -*- coding: utf-8 -*-
"""
多目标回归模型：参数搜索优化版，不输出 SHAP，自动保存最终完整参数 JPG

输出文件包括：
1. 训练集_带预测.csv
2. 测试集_带预测.csv
3. 多目标模型评估指标.csv
4. 多目标特征重要性.csv
5. 多目标特征重要性热力图.png
6. 多目标平均特征重要性.png
7. 模型名称_参数.jpg

核心功能：
1. 保留 MultiOutputRegressor 多目标逻辑
2. 使用 RandomizedSearchCV 在训练集内部自动调参
3. 使用较温和的参数范围，避免严重过拟合
4. 自动将每个模型最终实际使用的完整关键参数保存为 JPG 图片
"""

import os
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import xgboost as xgb
import lightgbm as lgb


# ==================== 0. 基础设置 ====================

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CLEAR_OLD_OUTPUT = True

CLIP_PREDICTIONS = True
Y_MIN = 1
Y_MAX = 10

N_ITER_SEARCH = 12
CV_FOLDS = 3
RANDOM_STATE = 42


# ==================== 1. 路径配置 ====================

input_file = r"D:\5.1test GVI\全部街景主观打分\3-四区主客观合并-有建筑15指标3000噪声删1000条.csv"

output_base = r"D:\5.1test GVI\全部街景主观打分\5.27多目标_参数搜索优化版"

xgb_output = os.path.join(output_base, "XGBoost")
rf_output = os.path.join(output_base, "随机森林")
gb_output = os.path.join(output_base, "Gradient Boosting")
lgbm_output = os.path.join(output_base, "LightGBM")

model_output_dirs = [xgb_output, rf_output, gb_output, lgbm_output]

if CLEAR_OLD_OUTPUT:
    for path in model_output_dirs:
        if os.path.exists(path):
            shutil.rmtree(path)

for path in [output_base] + model_output_dirs:
    os.makedirs(path, exist_ok=True)


# ==================== 2. 特征和目标 ====================

feature_cols = [
    'N01_绿视率', 'N02_水体', 'N03_天空', 'N04_自然地表',
    'B01_建筑与墙体', 'B02_障碍与桥隧',
    'T01_机动车空间', 'T02_机动车数量', 'T03_静态慢行设施',
    'T04_动态慢行参与者', 'T05_交通控制设施',
    'U01_休憩与自行车设施', 'U02_安全与照明设施',
    'U03_市政与便民设施', 'U04_界面信息杂乱度'
]

target_cols = ['安全', '活力', '美丽', '富裕', '无聊', '压抑']

id_cols = ['序号', '区域', 'image']


# ==================== 3. 数据读取 ====================

try:
    df = pd.read_csv(input_file, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(input_file, encoding='gbk')

df.columns = df.columns.astype(str).str.strip()

print("=" * 80)
print("数据读取成功")
print("数据行数：", len(df))
print("字段数量：", len(df.columns))
print("=" * 80)


# ==================== 4. 字段检查 ====================

missing_features = [col for col in feature_cols if col not in df.columns]
missing_targets = [col for col in target_cols if col not in df.columns]
missing_ids = [col for col in id_cols if col not in df.columns]

if missing_features:
    raise ValueError(f"缺少自变量字段：{missing_features}")

if missing_targets:
    raise ValueError(f"缺少目标变量字段：{missing_targets}")

if missing_ids:
    raise ValueError(f"缺少基础字段：{missing_ids}")


# ==================== 5. 数据清洗 ====================

for col in feature_cols + target_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

model_df = df[id_cols + feature_cols + target_cols].copy()

before_rows = len(model_df)
model_df = model_df.dropna(subset=feature_cols + target_cols)
after_rows = len(model_df)

print("删除缺失值前：", before_rows)
print("删除缺失值后：", after_rows)

X_full = model_df[feature_cols].copy()
Y_full = model_df[target_cols].copy()

print("X 维度：", X_full.shape)
print("Y 维度：", Y_full.shape)
print("Y 目标变量：", target_cols)


# ==================== 6. 分层划分训练集和测试集 ====================

try:
    stratify_label = pd.qcut(
        Y_full.mean(axis=1),
        q=5,
        labels=False,
        duplicates="drop"
    )

    if len(np.unique(stratify_label)) < 2:
        stratify_label = None
        print("分层标签不足，改用普通随机划分。")
    else:
        print("已启用分层抽样。")

except Exception as e:
    stratify_label = None
    print("分层抽样失败，改用普通随机划分：", e)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_full,
    Y_full,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=stratify_label
)

train_idx = Y_train.index
test_idx = Y_test.index

print("训练集数量：", len(X_train))
print("测试集数量：", len(X_test))


# ==================== 7. 多目标模型验证函数 ====================

def verify_multioutput_model(multi_model, Y_train_data, Y_test_pred):
    print("\n========== 多目标模型验证 ==========")
    print("Y_train 维度：", Y_train_data.shape)
    print("Y_train 列名：", Y_train_data.columns.tolist())
    print("模型类型：", type(multi_model))

    if hasattr(multi_model, "estimators_"):
        print("子模型数量：", len(multi_model.estimators_))

    print("Y_test_pred 维度：", Y_test_pred.shape)

    assert Y_train_data.shape[1] == len(target_cols), "错误：Y_train 不是 6 个目标变量"
    assert Y_test_pred.shape[1] == len(target_cols), "错误：预测结果不是 6 列"

    if hasattr(multi_model, "estimators_"):
        assert len(multi_model.estimators_) == len(target_cols), "错误：子模型数量不是 6 个"

    print("验证通过：当前代码是多目标 / 多输出回归逻辑。")
    print("====================================\n")


# ==================== 8. 模型评价函数 ====================

def evaluate_multioutput_model(model_name, dataset_name, Y_true, Y_pred):
    rows = []

    r2_each = r2_score(Y_true, Y_pred, multioutput='raw_values')
    mae_each = mean_absolute_error(Y_true, Y_pred, multioutput='raw_values')

    rmse_each = []
    for i, target in enumerate(target_cols):
        rmse = np.sqrt(mean_squared_error(Y_true.iloc[:, i], Y_pred[:, i]))
        rmse_each.append(rmse)

    for i, target in enumerate(target_cols):
        rows.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'Target': target,
            'R2': r2_each[i],
            'RMSE': rmse_each[i],
            'MAE': mae_each[i]
        })

    rows.append({
        'Model': model_name,
        'Dataset': dataset_name,
        'Target': '平均值',
        'R2': np.mean(r2_each),
        'RMSE': np.mean(rmse_each),
        'MAE': np.mean(mae_each)
    })

    return pd.DataFrame(rows)


# ==================== 9. 特征重要性函数 ====================

def get_multioutput_feature_importance(multi_model):
    importance_df = pd.DataFrame(index=feature_cols)

    for i, target in enumerate(target_cols):
        estimator = multi_model.estimators_[i]

        if hasattr(estimator, "feature_importances_"):
            importance_df[target] = estimator.feature_importances_
        else:
            importance_df[target] = np.nan

    importance_df["平均重要性"] = importance_df[target_cols].mean(axis=1)

    return importance_df


def plot_feature_importance_heatmap(importance_df, model_name, output_dir):
    data = importance_df[target_cols].T.values

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, aspect='auto')

    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=8)

    ax.set_yticks(np.arange(len(target_cols)))
    ax.set_yticklabels(target_cols, fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7)

    ax.set_title(f"{model_name} 多目标特征重要性热力图")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Importance")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "多目标特征重要性热力图.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def plot_average_importance(importance_df, model_name, output_dir):
    avg_importance = importance_df["平均重要性"].sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    avg_importance.plot(kind='barh')
    plt.xlabel("平均特征重要性")
    plt.ylabel("特征")
    plt.title(f"{model_name} 多目标平均特征重要性")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "多目标平均特征重要性.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


# ==================== 10. 保存最终完整参数为 JPG ====================

def get_final_params_for_display(model_name, multi_model):
    """
    从最终 best_estimator_ 中提取第一个子模型的实际参数。
    MultiOutputRegressor 内部 6 个子模型使用相同的最佳超参数。
    """

    estimator = multi_model.estimators_[0]
    params = estimator.get_params(deep=False)

    if "XGBoost" in model_name:
        param_order = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
            "objective",
            "tree_method",
            "random_state",
            "n_jobs"
        ]

    elif "随机森林" in model_name:
        param_order = [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "max_samples",
            "criterion",
            "random_state",
            "n_jobs"
        ]

    elif "Gradient Boosting" in model_name:
        param_order = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "loss",
            "random_state"
        ]

    elif "LightGBM" in model_name:
        param_order = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "num_leaves",
            "subsample",
            "colsample_bytree",
            "min_child_samples",
            "reg_alpha",
            "reg_lambda",
            "objective",
            "random_state",
            "n_jobs",
            "verbose"
        ]

    else:
        param_order = list(params.keys())

    display_params = []

    for key in param_order:
        if key in params:
            display_params.append({
                "参数名称": key,
                "最终取值": params[key]
            })

    return pd.DataFrame(display_params)


def save_final_params_as_jpg(model_name, best_cv_r2, multi_model, output_dir):
    """
    将每个模型最终实际使用的完整关键参数保存为 JPG 图片。
    文件名格式：模型名称_参数.jpg
    """

    safe_model_name = (
        model_name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
        .replace(" ", "_")
    )

    params_df = get_final_params_for_display(model_name, multi_model)

    fig_height = max(5, 0.45 * len(params_df) + 2.5)

    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    title = f"{model_name} 最终使用参数设置"
    subtitle = f"Best CV R2 = {best_cv_r2:.6f}"

    ax.text(
        0.5, 0.96,
        title,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        transform=ax.transAxes
    )

    ax.text(
        0.5, 0.90,
        subtitle,
        ha="center",
        va="top",
        fontsize=13,
        transform=ax.transAxes
    )

    table = ax.table(
        cellText=params_df.values,
        colLabels=params_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.55, 0.35]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)

        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E6E6E6")
        else:
            cell.set_facecolor("white")

    jpg_path = os.path.join(output_dir, f"{safe_model_name}_参数.jpg")

    plt.tight_layout()
    plt.savefig(
        jpg_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        format="jpg"
    )
    plt.close()

    print(f"{model_name} 最终完整参数 JPG 已保存：{jpg_path}")


# ==================== 11. 保存预测结果函数 ====================

def save_prediction_results(output_dir, Y_train_pred, Y_test_pred):
    train_result = model_df.loc[train_idx, id_cols + feature_cols + target_cols].copy()
    test_result = model_df.loc[test_idx, id_cols + feature_cols + target_cols].copy()

    for i, target in enumerate(target_cols):
        train_result[f'pred_{target}'] = Y_train_pred[:, i]
        test_result[f'pred_{target}'] = Y_test_pred[:, i]

    train_result.to_csv(
        os.path.join(output_dir, "训练集_带预测.csv"),
        index=False,
        encoding='utf-8-sig'
    )

    test_result.to_csv(
        os.path.join(output_dir, "测试集_带预测.csv"),
        index=False,
        encoding='utf-8-sig'
    )


# ==================== 12. 参数搜索函数 ====================

def tune_multioutput_model(model_name, base_model, param_distributions, output_dir):
    """
    在训练集内部进行参数搜索。
    注意：这里只用训练集做交叉验证，不碰测试集。
    同时保存最终实际使用的完整参数为 JPG。
    """

    print("\n" + "-" * 80)
    print(f"开始参数搜索：{model_name}")
    print("-" * 80)

    multi_model = MultiOutputRegressor(
        estimator=base_model,
        n_jobs=1
    )

    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        estimator=multi_model,
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        scoring='r2',
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=2
    )

    search.fit(X_train, Y_train)

    print(f"\n{model_name} 最佳交叉验证 R2：", search.best_score_)
    print(f"{model_name} 最佳搜索参数：")
    print(search.best_params_)

    best_model = search.best_estimator_

    save_final_params_as_jpg(
        model_name=model_name,
        best_cv_r2=search.best_score_,
        multi_model=best_model,
        output_dir=output_dir
    )

    return best_model


# ==================== 13. 多目标模型运行函数 ====================

def run_multioutput_model(model_name, output_dir, base_model, param_distributions):
    print("\n" + "=" * 80)
    print(f"开始训练多目标模型：{model_name}")
    print("=" * 80)

    multi_model = tune_multioutput_model(
        model_name=model_name,
        base_model=base_model,
        param_distributions=param_distributions,
        output_dir=output_dir
    )

    Y_train_pred = multi_model.predict(X_train)
    Y_test_pred = multi_model.predict(X_test)

    if CLIP_PREDICTIONS:
        Y_train_pred = np.clip(Y_train_pred, Y_MIN, Y_MAX)
        Y_test_pred = np.clip(Y_test_pred, Y_MIN, Y_MAX)

    verify_multioutput_model(
        multi_model=multi_model,
        Y_train_data=Y_train,
        Y_test_pred=Y_test_pred
    )

    save_prediction_results(
        output_dir=output_dir,
        Y_train_pred=Y_train_pred,
        Y_test_pred=Y_test_pred
    )

    train_metrics = evaluate_multioutput_model(
        model_name=model_name,
        dataset_name="Train",
        Y_true=Y_train,
        Y_pred=Y_train_pred
    )

    test_metrics = evaluate_multioutput_model(
        model_name=model_name,
        dataset_name="Test",
        Y_true=Y_test,
        Y_pred=Y_test_pred
    )

    metrics_df = pd.concat([train_metrics, test_metrics], ignore_index=True)

    metrics_df.to_csv(
        os.path.join(output_dir, "多目标模型评估指标.csv"),
        index=False,
        encoding='utf-8-sig'
    )

    importance_df = get_multioutput_feature_importance(multi_model)

    importance_df.to_csv(
        os.path.join(output_dir, "多目标特征重要性.csv"),
        encoding='utf-8-sig'
    )

    plot_feature_importance_heatmap(
        importance_df=importance_df,
        model_name=model_name,
        output_dir=output_dir
    )

    plot_average_importance(
        importance_df=importance_df,
        model_name=model_name,
        output_dir=output_dir
    )

    print(f"{model_name} 训练完成，结果保存到：{output_dir}")

    return multi_model, metrics_df, importance_df


# ==================== 14. 基础模型与参数搜索范围 ====================

xgb_base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

xgb_param_distributions = {
    'estimator__n_estimators': [650, 750, 850, 950, 1100],
    'estimator__max_depth': [3, 4],
    'estimator__learning_rate': [0.025, 0.03, 0.035, 0.04],
    'estimator__subsample': [0.8, 0.85, 0.9],
    'estimator__colsample_bytree': [0.8, 0.85, 0.9],
    'estimator__min_child_weight': [6, 8, 10],
    'estimator__gamma': [0, 0.05, 0.1],
    'estimator__reg_alpha': [0.1, 0.3, 0.5],
    'estimator__reg_lambda': [8, 10, 12, 15]
}

rf_base_model = RandomForestRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_param_distributions = {
    'estimator__n_estimators': [600, 800, 1000],
    'estimator__max_depth': [10, 12, 14],
    'estimator__min_samples_split': [5, 8, 10],
    'estimator__min_samples_leaf': [2, 3, 4],
    'estimator__max_features': [0.7, 0.8, 0.9],
    'estimator__bootstrap': [True],
    'estimator__max_samples': [0.75, 0.8, 0.85]
}

gb_base_model = GradientBoostingRegressor(
    random_state=RANDOM_STATE
)

gb_param_distributions = {
    'estimator__n_estimators': [450, 550, 650, 750],
    'estimator__max_depth': [2, 3, 4],
    'estimator__learning_rate': [0.025, 0.03, 0.035, 0.04],
    'estimator__subsample': [0.75, 0.8, 0.85],
    'estimator__min_samples_split': [5, 8, 10],
    'estimator__min_samples_leaf': [2, 3, 4],
    'estimator__max_features': [0.7, 0.8, 0.9]
}

lgbm_base_model = lgb.LGBMRegressor(
    objective='regression',
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)

lgbm_param_distributions = {
    'estimator__n_estimators': [650, 750, 850, 950, 1100],
    'estimator__max_depth': [4, 5, 6],
    'estimator__learning_rate': [0.025, 0.03, 0.035, 0.04],
    'estimator__num_leaves': [15, 20, 25, 31],
    'estimator__subsample': [0.8, 0.85, 0.9],
    'estimator__colsample_bytree': [0.8, 0.85, 0.9],
    'estimator__min_child_samples': [15, 20, 25, 30],
    'estimator__reg_alpha': [0.1, 0.3, 0.5],
    'estimator__reg_lambda': [8, 10, 12, 15]
}


# ==================== 15. 运行四个多目标模型 ====================

xgb_model, xgb_metrics, xgb_importance = run_multioutput_model(
    model_name="XGBoost_多目标",
    output_dir=xgb_output,
    base_model=xgb_base_model,
    param_distributions=xgb_param_distributions
)

rf_model, rf_metrics, rf_importance = run_multioutput_model(
    model_name="随机森林_多目标",
    output_dir=rf_output,
    base_model=rf_base_model,
    param_distributions=rf_param_distributions
)

gb_model, gb_metrics, gb_importance = run_multioutput_model(
    model_name="Gradient Boosting_多目标",
    output_dir=gb_output,
    base_model=gb_base_model,
    param_distributions=gb_param_distributions
)

lgbm_model, lgbm_metrics, lgbm_importance = run_multioutput_model(
    model_name="LightGBM_多目标",
    output_dir=lgbm_output,
    base_model=lgbm_base_model,
    param_distributions=lgbm_param_distributions
)


# ==================== 16. 完成提示 ====================

print("\n" + "=" * 80)
print("全部多目标模型参数搜索与训练完成！")
print("=" * 80)

print("\n每个模型文件夹输出以下文件：")
print("1. 训练集_带预测.csv")
print("2. 测试集_带预测.csv")
print("3. 多目标模型评估指标.csv")
print("4. 多目标特征重要性.csv")
print("5. 多目标特征重要性热力图.png")
print("6. 多目标平均特征重要性.png")
print("7. 模型名称_参数.jpg")

print("\n输出文件夹：")
print("XGBoost：", xgb_output)
print("随机森林：", rf_output)
print("Gradient Boosting：", gb_output)
print("LightGBM：", lgbm_output)
