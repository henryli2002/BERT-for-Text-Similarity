import warnings
import openai
import os
import pandas as pd

openai.api_key = 'YOUR_OPENAI_API_KEY'

MODEL_TYPE = "gpt-3.5-turbo-0613" # 

def analyze_text_similarity(text1, text2, model="gpt-3.5-turbo-0613"):
    results = []
    for t1, t2 in zip(text1, text2):
        prompt = f"""
        文本A: {t1}
        文本B: {t2}
        请评估文本A和文本B之间的相关性，并给出一个0到5之间的分数，表示它们的相关程度。5表示完全一致，0表示无关。以下有几个实例供你参考：\
            “一架飞机要起飞了。	一架飞机正在起飞。	5”\
            “一个人在拉大提琴。	一个坐着的人正在拉大提琴。	4”\
            “一个男人在吹一支大笛子。	一个人在吹长笛。	3”\
            “三个人在下棋。	两个人在下棋。	2”\
            “一个人在切西红柿。	一个人在切肉。	1”\
            “一个男人在抽烟。	一个男人在滑冰。	0”，你只需给出单个0-5的整数数字分数组成的列表即可，以便我进行统计
        """
        request = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": " "
                },
            ]
        }

        try:
            response = openai.ChatCompletion.create(
                model=request["model"],
                messages=request["messages"]
            )
            model_response = response.choices[0].message["content"]
            results.append(model_response)
        except Exception as e:
            results.append(f"OpenAI 接口调用出错: {e}")

    return results


def save_results_to_txt(text1, text2, labels, results, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        for t1, t2, label, score in zip(text1, text2, labels, results):
            file.write(f"Text1: {t1}\t")
            file.write(f"Text2: {t2}\t")
            file.write(f"Label: {label}\t")
            file.write(f"Score: {score}\t\n")


if __name__ == "__main__":
    # column_names = ['text1', 'text2', 'labels']
    # df_test = pd.read_csv('Chinese-STS-B/sts-b-test.txt', sep='\t', names=column_names)
    # text1 = df_test['text1'].values.tolist()
    # text2 = df_test['text2'].values.tolist()
    # labels = df_test['labels'].values.tolist()

    text1 = ["今天天气晴朗，阳光明媚。", "我喜欢读书学习。"]
    text2 = ["天空湛蓝，阳光明媚，没有一丝云彩。", "我不喜欢读书，更喜欢户外运动。"]
    labels = ["", "不相关"]

    results = analyze_text_similarity(text1, text2, model=MODEL_TYPE)

    # 保存结果到文件
    filepath = os.path.join('./output', 'gpt_output.txt')
    save_results_to_txt(text1, text2, labels, results, filepath)

    print(f"结果已保存到 {filepath}")

import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

def fine_tune_openai_model(text_pairs, labels, model="gpt-3.5-turbo-0613"):
    training_data = []
    for text_pair, label in zip(text_pairs, labels):
        training_data.append({
            "text": f"文本A: {text_pair[0]} 文本B: {text_pair[1]}",
            "label": label
        })

    fine_tune_request = {
        "model": model,
        "train_data": training_data,
        "train_batch_size": 32,
        "train_epochs": 3
    }

    try:
        response = openai.FineTune.create(**fine_tune_request)
        fine_tuned_model_id = response.model.id
        return fine_tuned_model_id
    except Exception as e:
        print(f"OpenAI Fine-Tune API调用出错: {e}")
        return None

if __name__ == "__main__":
    text_pairs = [
        ("今天天气晴朗，阳光明媚。", "天空湛蓝，阳光明媚，没有一丝云彩。"),
        ("我喜欢读书学习。", "我不喜欢读书，更喜欢户外运动。")
    ]
    labels = ["相关", "不相关"]

    fine_tuned_model_id = fine_tune_openai_model(text_pairs, labels, model=MODEL_TYPE)
    if fine_tuned_model_id:
        print(f"微调完成，微调后的模型ID为: {fine_tuned_model_id}")
    else:
        print("微调失败，请检查输入数据并重试。")
