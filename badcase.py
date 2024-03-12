import pandas as pd

document = './output/bert_output'

# 加载数据
df = pd.read_csv(f'{document}.txt', sep='\t')

# 重新计算accuracy
accuracy = (df['labels'] == df['predictions']).mean()
print(f"Accuracy: {accuracy:.4f}")

# 计算bad case率
# 定义bad case为标签和预测之间的差异绝对值大于1
df['bad_case'] = (df['labels'] - df['predictions']).abs() > 1
bad_case_rate = df['bad_case'].mean()
print(f"Bad case rate: {bad_case_rate:.4f}")

# 筛选出bad cases
bad_cases = df[df['bad_case']]

# 将bad cases保存到新文件
bad_cases.to_csv(f'{document}_badcase.txt', sep='\t', index=False, header=True)

# 打印bad cases
print(f"Bad cases saved to {document}_badcase.txt")
