import os 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------- 路径配置 --------------------
input_file = r"D:\5.1test GVI\全部街景主观打分\3-四区主客观合并-有建筑15指标3000噪声删1000条.csv"
output_base = r"D:\5.1test GVI\全部街景主观打分\5.11预测"
rf_output = os.path.join(output_base, "随机森林")
xgb_output = os.path.join(output_base, "XGBoost")
os.makedirs(rf_output, exist_ok=True)
os.makedirs(xgb_output, exist_ok=True)

# -------------------- 特征和目标 --------------------
feature_cols = [
    'N01_绿视率','N02_水体','N03_天空','N04_自然地表',
    'B01_建筑与墙体','B02_障碍与桥隧',
    'T01_机动车空间','T02_机动车数量','T03_静态慢行设施','T04_动态慢行参与者','T05_交通控制设施',
    'U01_休憩与自行车设施','U02_安全与照明设施','U03_市政与便民设施','U04_界面信息杂乱度'
]
target_cols = ['安全','活力','美丽','富裕','无聊','压抑']

# -------------------- 数据读取 --------------------
try:
    df = pd.read_csv(input_file, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(input_file, encoding='gbk')

X_full = df[feature_cols].copy()
y_full = df[target_cols].copy()

# -------------------- 划分训练/测试集 --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)
train_idx = y_train.index
test_idx = y_test.index

# -------------------- 模型训练评估函数 --------------------
def train_evaluate_model(model, X_tr, X_te, y_tr, y_te, target):
    model.fit(X_tr, y_tr[target])
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    rmse_te = np.sqrt(mean_squared_error(y_te[target], y_pred_te))
    metrics = {
        'Target': target,
        'Train_R2': r2_score(y_tr[target], y_pred_tr),
        'Test_R2': r2_score(y_te[target], y_pred_te),
        'Test_RMSE': rmse_te,
        'Test_MAE': mean_absolute_error(y_te[target], y_pred_te)
    }
    imp = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    return model, metrics, y_pred_tr, y_pred_te, imp

# -------------------- 单因子 SHAP 散点图函数 --------------------
def save_single_factor_shap(shap_values, X, target, model_name, model_output_dir):
    target_dir = os.path.join(model_output_dir, f"{target}_单因子图")
    os.makedirs(target_dir, exist_ok=True)
    for i, feature in enumerate(feature_cols):
        plt.figure(figsize=(6,4))
        plt.scatter(X[feature], shap_values[:, i], alpha=0.6)
        plt.xlabel(feature)
        plt.ylabel("SHAP value")
        plt.title(f"{model_name} {target} - {feature}")
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, f"{feature}_SHAP.png"), dpi=300)
        plt.close()

# ===================== 随机森林 =====================
rf_models = {}
rf_metrics_list = []
rf_feature_importance = pd.DataFrame(index=feature_cols)
rf_train_pred = pd.DataFrame(index=y_train.index)
rf_test_pred = pd.DataFrame(index=y_test.index)

for target in target_cols:
    rf = RandomForestRegressor(
        n_estimators=1800, max_depth=18, min_samples_split=3, min_samples_leaf=1,
        max_features=0.9, bootstrap=True, max_samples=0.85,
        random_state=42, n_jobs=-1
    )
    model, metrics, y_pred_tr, y_pred_te, imp = train_evaluate_model(rf, X_train, X_test, y_train, y_test, target)
    rf_models[target] = model
    rf_metrics_list.append(metrics)
    rf_feature_importance[target] = imp
    rf_train_pred[target] = y_pred_tr
    rf_test_pred[target] = y_pred_te

# 保存训练/测试集预测结果
df_train_pred = df.loc[train_idx, ['序号','区域','image'] + feature_cols + target_cols].copy()
df_test_pred = df.loc[test_idx, ['序号','区域','image'] + feature_cols + target_cols].copy()
for t in target_cols:
    df_train_pred[f'pred_{t}'] = rf_train_pred[t].values
    df_test_pred[f'pred_{t}'] = rf_test_pred[t].values
df_train_pred.to_csv(os.path.join(rf_output, "训练集_带预测.csv"), index=False, encoding='utf-8-sig')
df_test_pred.to_csv(os.path.join(rf_output, "测试集_带预测.csv"), index=False, encoding='utf-8-sig')

# 保存特征重要性和模型评估
rf_feature_importance.to_csv(os.path.join(rf_output, "特征重要性.csv"), encoding='utf-8-sig')
pd.DataFrame(rf_metrics_list).to_csv(os.path.join(rf_output, "模型评估指标.csv"), index=False, encoding='utf-8-sig')

# 特征重要性热力图
plt.figure(figsize=(12,6))
sns.heatmap(rf_feature_importance.T, annot=True, fmt=".3f", cmap='YlOrRd', cbar_kws={'label':'Importance'})
plt.title("随机森林特征重要性热力图")
plt.tight_layout()
plt.savefig(os.path.join(rf_output, "feature_importance_heatmap.png"), dpi=300)
plt.close()

# SHAP summary, bar, 单因子图
for target in target_cols:
    explainer_rf = shap.TreeExplainer(rf_models[target])
    shap_values_rf = explainer_rf.shap_values(X_train)
    # summary plot
    shap.summary_plot(shap_values_rf, X_train, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(rf_output, f"{target}_SHAP_summary.png"), dpi=300)
    plt.close()
    # bar plot
    shap.summary_plot(shap_values_rf, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(rf_output, f"{target}_SHAP_bar.png"), dpi=300)
    plt.close()
    # 单因子图
    save_single_factor_shap(shap_values_rf, X_train, target, "随机森林", rf_output)

# ===================== XGBoost =====================
xgb_models = {}
xgb_metrics_list = []
xgb_feature_importance = pd.DataFrame(index=feature_cols)
xgb_train_pred = pd.DataFrame(index=y_train.index)
xgb_test_pred = pd.DataFrame(index=y_test.index)

for target in target_cols:
    xgb_model = xgb.XGBRegressor(
        n_estimators=1800, max_depth=7, learning_rate=0.02,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=4, reg_lambda=5,
        objective='reg:squarederror', tree_method='hist',
        random_state=42, n_jobs=-1
    )
    model, metrics, y_pred_tr, y_pred_te, imp = train_evaluate_model(xgb_model, X_train, X_test, y_train, y_test, target)
    xgb_models[target] = model
    xgb_metrics_list.append(metrics)
    xgb_feature_importance[target] = imp
    xgb_train_pred[target] = y_pred_tr
    xgb_test_pred[target] = y_pred_te

# 保存训练/测试集预测结果
df_train_pred_xgb = df.loc[train_idx, ['序号','区域','image'] + feature_cols + target_cols].copy()
df_test_pred_xgb = df.loc[test_idx, ['序号','区域','image'] + feature_cols + target_cols].copy()
for t in target_cols:
    df_train_pred_xgb[f'pred_{t}'] = xgb_train_pred[t].values
    df_test_pred_xgb[f'pred_{t}'] = xgb_test_pred[t].values
df_train_pred_xgb.to_csv(os.path.join(xgb_output, "训练集_带预测.csv"), index=False, encoding='utf-8-sig')
df_test_pred_xgb.to_csv(os.path.join(xgb_output, "测试集_带预测.csv"), index=False, encoding='utf-8-sig')

# 保存特征重要性和模型评估
xgb_feature_importance.to_csv(os.path.join(xgb_output, "特征重要性.csv"), encoding='utf-8-sig')
pd.DataFrame(xgb_metrics_list).to_csv(os.path.join(xgb_output, "模型评估指标.csv"), index=False, encoding='utf-8-sig')

# 特征重要性热力图
plt.figure(figsize=(12,6))
sns.heatmap(xgb_feature_importance.T, annot=True, fmt=".3f", cmap='YlOrRd', cbar_kws={'label':'Importance'})
plt.title("XGBoost特征重要性热力图")
plt.tight_layout()
plt.savefig(os.path.join(xgb_output, "feature_importance_heatmap.png"), dpi=300)
plt.close()

# SHAP summary, bar, 单因子图
for target in target_cols:
    explainer_xgb = shap.TreeExplainer(xgb_models[target])
    shap_values_xgb = explainer_xgb.shap_values(X_train)
    # summary plot
    shap.summary_plot(shap_values_xgb, X_train, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(xgb_output, f"{target}_SHAP_summary.png"), dpi=300)
    plt.close()
    # bar plot
    shap.summary_plot(shap_values_xgb, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(xgb_output, f"{target}_SHAP_bar.png"), dpi=300)
    plt.close()
    # 单因子图
    save_single_factor_shap(shap_values_xgb, X_train, target, "XGBoost", xgb_output)
