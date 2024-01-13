from scipy.stats import expon

# 均值 方差 偏度（三阶中心距） 峰度（四阶中心距）
print(expon.stats(scale=3, moments='mvsk'))