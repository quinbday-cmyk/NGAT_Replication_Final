The original paper is [NGAT: A Node-level Graph Attention Network
for Long-term Stock Prediction](https://arxiv.org/pdf/2507.02018), their GitHub repository is located [here](https://github.com/FreddieNIU/NGAT/tree/main).

The two data sets, [ACL2018](https://github.com/yumoxu/stocknet-dataset) (2014-2016) and [SPNews](https://github.com/FreddieNIU/Financial-Graph-Evaluation/tree/main) (2022-2024) originally were for [Stock Movement Prediction from Tweets and Historical Prices](https://aclanthology.org/P18-1183/) and [Evaluating Financial Relational Graphs: Interpretation Before Prediction](https://dl.acm.org/doi/abs/10.1145/3677052.3698644) respectively.


I am unsure if the model is local or not. If it is not on your computer, please run main.py, predict.py, main_classification.py, data_interpreter.ipynb, and output_review.ipynb to obtain all necessary files and setups to run Final_Report.ipynb.

If the model is not local, it need to be trained for standard deviation for both data sets through main_classification.py for the full analysis in Final_Project.ipynb to work. New Data Generation is new data for all 88 companies in ACL2018 until 02/06/2026 to attempt to apply either ACL2018 or SPNews to the data to predict it, although it also needs to be trained since my computer has low memory.
