# -*- coding:utf-8 -*-
"""
@author : wang bq
@email  : 
@time   :20-12-4 下午3:44
@IDE    :PyCharm
@document   :main.py
"""

from spo_ie.inference import IeModel

if __name__ == "__main__":

    model = IeModel()

    # text = '《离开》是由张宇谱曲，演唱'
    # text = '焦作市河阳酒精实业有限公司于2001年2月23日在孟州市工商行政管理局登记成立'
    text = '马克西·莫拉雷斯（Maximiliano Morales），男，1987年02月27日出生，阿根廷足球运动员，身高1米60，体重53公斤，绰号“小巨人”，是一名足球运动员，2011年加盟意大利亚特兰大队'
    res = model.output(text)
    print(res)
