# ElementImportance
融合空间特征的灾害要素重要性评估

此工作是作为地理信息可视化的前一步工作，为了根据灾害场景中不同空间要素的重要性差异而进行可视化增强

基于数据集：RescueNet (参考文献：Rahnemoonfar, M., Chowdhury, T. & Murphy, R. RescueNet: A High Resolution UAV Semantic Segmentation Dataset for Natural Disaster Damage Assessment. Sci Data 10, 913 (2023). https://doi.org/10.1038/s41597-023-02799-4）

这个数据集有较为完整的灾害场景，不足之处是没有真实的空间参考，所以只能通过像素坐标来计算，但原理是一样的，空间密度和空间距离都是存在的。

DIE
在Reduce/VecVisualize.py中写好可以批量可视化Reduce/output/vectors.json矢量，然后输出到output，命名规则（id_vec.png)，颜色文本（"D:\ElementImportance\Data\QGIS_label_style.txt"）