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

项目扩展库：rsa + pycryptodome

项目结构如下

* models

  Client：定义联邦学习客户端

  models：存放了BiLSTM，CNN_LSTM，CNN-2 layers-LSTM网络模型

  Server：定义联邦学习服务端

  Test：进行联邦学习架构最后的模型测试

* model_selection

  client_train：查看50个用户在BiLSTM上的运行效果

  model_select：通过比较BiLSTM，CNN_LSTM，双层CNN_LSTM，选择BiLSTM作为最终预测模型

  model_test：进行测试集性能测试

  model_train：进行训练集训练

* network

  存放了联邦学习每轮epoch，中心发放的网络结构

* source_data

  存放了112个不同block的电力数据

* utils

  aes_algo：定义AES加密算法

  data_process：进行数据的预处理

  options：存放了所有模型训练的参数

  parameter_tran：提供将所有的网络参数转成可加密格式、将字符串转成网络参数格式等方法

  rsa_algo：定义RSA加密算法
  
  ---
  
  model_avg：进行联邦学习运算，为本项目的主要运行代码
  
  model_contrast：对比没有使用联邦学习的训练效果
  
  result_show：展示联邦学习训练中，在时序图中的具体效果

Tips：本次代码重构，抽象出Client、Server，并利用观察者模式，将model_avg与Server解耦，增强了代码的可读性和维护的方便性
