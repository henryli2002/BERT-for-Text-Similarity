# BERT-for-Text-Similarity
BUPT智能信息网络实验，本实验为课程小实验4个中第4个
BERT-FOR-TEXT-SIMILARITY/
│
├── Chinese-STS-B/              # 数据集
│   ├── sts-b-dev.txt
│   ├── sts-b-test.txt
│   └── sts-b-train.txt
│
├── models/   
│   ├── model_0e.pth            # 模型参数(1epoch)
│   ...            
│   
│
├── output/                     # 结果产出
│   ├── bert_output_badcase.txt
│   └── bert_output.txt
│
├── pic/                        # 可视化分析图像
│   ├── Confusion_Matrix_2.png
│   ├── Confusion_Matrix_6.png
│   ├── Distribution_2.png 
│   └── Distribution_2.png
│
├── .gitignore                  # 指定git忽略的文件和目录
├── README.md                   # 项目介绍和说明文件
├── badcase.py                  # 找出badcase的脚本
├── bert_model.py               # bert模型的训练验证和测试
└── requirements.txt            # 项目依赖
