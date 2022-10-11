# Multi-Source-Electricity-Market-Forecast

The project is based on a time-series model combined with a federal learning framework for electricity load forecasting in multi-source electricity markets.

And we will build the final model by conducting differtent routes. Here are the differtent routes.

* Data Collection & Background Research
* Data Preprocessing & Feature Engineering
* Build an electricity load forecasting model 
* Federated Learning Framework on Privicy
* Combining Federated Learning and Forecastring models
* Model Synthesis Analysis & Issue a paper

This project officially started in early September 2022 and will be continuously updated and optimized.

项目环境：Python3.9 + Pytorch + Cuda 11.3

项目结构如下

* model_selection

  client_train：查看50个用户在BiLSTM上的运行效果

  model_select：通过比较BiLSTM，CNN_LSTM，双层CNN_LSTM，选择BiLSTM作为最终预测模型

  model_test：进行测试集性能测试

  model_train：进行训练集训练

* models

  Fedavg：网络参数Avg算法

  models：定义了三种网络结构BiLSTM，CNN_LSTM、CNN_LSTM_2

  Test：进行联邦学习架构最后的模型测试

  Update：进行联邦学习本地client的模型更新

* network

  存放了联邦学习每轮epoch，中心发放的网络结构

* source_data

  存放了112个不同block的电力数据

* utils

  data_process：进行数据的预处理

  options：存放了所有模型训练的参数

  ---

  model_avg：进行联邦学习运算，为本项目的主要运行代码

  model_contrast：对比没有使用联邦学习的训练效果
  
  result_show：展示联邦学习训练中，在时序图中的具体效果
