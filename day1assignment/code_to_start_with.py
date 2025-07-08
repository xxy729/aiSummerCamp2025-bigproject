
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 加载数据
data = pd.read_csv('data/train.csv')  # 确保train.csv与代码文件在同一目录，或修改为实际路径
df = data.copy()

# 数据预处理
# 删除无关特征
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# 处理缺失值
print("处理前缺失值情况：")
print(df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)  # 用中位数填充年龄缺失值
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # 用众数填充登船港口缺失值
print("\n处理后缺失值情况：")
print(df.isnull().sum())

# 转换分类数据为数值型
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 分离特征和标签
X = df.drop('Survived', axis=1)
y = df['Survived']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建模型
# 1. 支持向量机(SVM)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 2. K近邻(KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# 3. 随机森林(Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # 随机森林不需要特征标准化

# 预测与评估
models = {
    'SVM': svm_model,
    'KNN': knn_model,
    'Random Forest': rf_model
}

print("\n模型评估结果：")
for name, model in models.items():
    if name == 'Random Forest':
        y_pred = model.predict(X_test)
    else:
        y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} 准确率: {accuracy:.4f}")
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    print("分类报告:")
    print(classification_report(y_test, y_pred))

# 选择表现最好的模型进行最终预测（这里选择随机森林作为示例）
best_model = rf_model
final_pred = best_model.predict(X_test)

# 输出部分预测结果
print("\n部分预测结果（实际值 vs 预测值）：")
comparison = pd.DataFrame({'实际存活': y_test, '预测存活': final_pred})
print(comparison.head(10))