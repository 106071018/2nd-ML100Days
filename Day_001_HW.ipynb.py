#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###作業1：
請上 Kaggle, 在 Competitions 或 Dataset 中找一組競賽或資料並寫下：
1. 你選的這組資料為何重要
#因為本身為財金相關科系，選一個自己有興趣的資料，初探新聞報導對於股價的波動影響，若新聞可影響股市股價的話，希望可以找到其中關聯。
2. 資料從何而來 (tips: 譬如提供者是誰、以什麼方式蒐集)
#競賽由Two Sigma提供，而資料的收集方式廣泛，可以以美國的財報、財經新聞報導的特定股價作為實驗對象（希望data數量夠多），再由美國股市的當日（假設股市投資人為平均理性，事件會瞬間反應在股價上）
#為參考數據，以此去計算相關係數。
#Market data provided by Intrinio
#News data provided by Thomson Reuters
3. 蒐集而來的資料型態為何
#為結構化資料，大量的股市數據
4. 這組資料想解決的問題如何評估
#評估MAP
作業2：
想像你經營一個自由載客車隊，你希望能透過數據分析以提升業績，請你思考並描述你如何規劃整體的分析/解決方案：
1. 核心問題為何 (tips：如何定義 「提升業績 & 你的假設」)
#提升業績、假設：將總營業額提升20%
#以人手是可以應付市場需求、營業地區固定（此假設為台北市）為前提。
2. 資料從何而來 (tips：哪些資料可能會對你想問的問題產生影響 & 資料如何蒐集)
#需要資料（以上皆分區統計）：
(1.)台北市各時段的非自駕人數（指不使用自己的車）
(2.)台北市市民的計程車滿意度等相關調查資料
(3.)台北市市民的計程車需求量
3. 蒐集而來的資料型態為何
(1.)結構化資料：數值、表格。
(2.)非結構化資料：文字。
4. 你要回答的問題，其如何評估 (tips：你的假設如何驗證)
#將每個區域的需求做統計，可知每個區需要分派多少人力，讓人力不過度集中，避免互搶業績同時提升每個人力的生產力。
#將台北市各時段的非自駕的人數做統計，可知在每個區域中的以分派人力，再次進行分派，以需求的高峰、低谷作比例分配，提高效率。
#以台北市民的乘車滿意度做統計，了解消費者喜好，在車隊服務上提高服務品質，從而讓顧客對車隊的滿意度提升，增加回客率。
#以台北市民的電話叫車習慣做統計，了解消費者對於叫車服務的使用比例，可進行相關轎車服務的活動（折扣），使消費者養成叫自家車隊的習慣，提高熟客比例。

###將以上幾點做改善調整後，以一季為檢視單位，觀察營業額的變化同時審視改善短處，增其長處，相信對公司是一大成長。
作業3：
請點選下方檢視範例依照 Day_001_example_of_metrics.ipynb 完成 Mean Squared Error 的函式
資料夾結構建立規則提醒：2nd-ML100Days > data資料夾 & homework資料夾 (ipynb檔) 
(請注意data的存放位置，建議放在*.ipynb 同一個目錄下，這樣才能在不修改code的情況下正常執行)
###


def mean_absolute_error(y, yp):
    """
    計算 MAE
    Args:
        - y: 實際值
        - yp: 預測值
    Return:
        - mae: MAE
    """
    mae = MAE = sum(abs(y - yp)) / len(y)
    return mae

MAE = mean_absolute_error(y, y_hat)
print("The Mean absolute error is %.3f" % (MAE))


    


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
def mean_square_error(x,xp):
    mse=MSE=sum(x-xp)**2/len(x)


# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


w = 3
b = 0.5

x_lin = np.linspace(0, 100, 101)

y = (x_lin + np.random.randn(101) * 5) * w + b

plt.plot(x_lin, y, 'b.', label = 'data points')
plt.title("Assume we have data points")
plt.legend(loc = 2)
plt.show()


# In[3]:


y_hat = x_lin * w + b
plt.plot(x_lin, y, 'b.', label = 'data')
plt.plot(x_lin, y_hat, 'r-', label = 'prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc = 2)
plt.show()


# In[4]:


def mean_absolute_error(y, yp):
    """
    計算 MAE
    Args:
        - y: 實際值
        - yp: 預測值
    Return:
        - mae: MAE
    """
    mae = MAE = sum(abs(y - yp)) / len(y)
    return mae

MAE = mean_absolute_error(y, y_hat)
print("The Mean absolute error is %.3f" % (MAE))


# In[5]:


def mean_square_error(y,yp):
    mse=MSE=sum((y-yp)**2)/len(y)
    return mse
MSE=mean_square_error(y,y_hat)
print("The Mean square error is %.3f"%(MSE))


# In[ ]:




