import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


class FOG:
    start = 0
    start_fog = 0
    end = 0
    end_fog = 0
    data = pd.DataFrame([])
    file_name = ""

    def __init__(self, start, end):
        # print('A new fog item')
        self.start_fog = start
        self.end_fog = end
        self.start = start - 64 * 10
        self.end = end

    def __repr__(self):
        return "start:{}, end:{}, data:{}".format(self.start_fog, self.end_fog, self.data.shape)

    def set_data(self, data):
        self.data = data

    # def show_fog_data(self, title=""):
    #     fog1_data = self.data
    #     x = range(0, fog1_data.shape[0])
    #     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    #     plt.xlabel(title)
    #     mx_y = np.max(np.max(fog1_data.loc[:, ["ankle_vertical", "ankle_hori_fw", "ankle_hori_lat"]]))
    #     mn_y = np.min(np.min(fog1_data.loc[:, ["ankle_vertical", "ankle_hori_fw", "ankle_hori_lat"]]))
    #     ax1.fill_betweenx(y=[mn_y, mx_y], x1=self.start_fog-self.start, facecolor='black', alpha=0.5)
    #     ax1.plot(x, fog1_data['ankle_vertical'], 'r')
    #     ax1.plot(x, fog1_data['ankle_hori_fw'], 'b')
    #     ax1.plot(x, fog1_data['ankle_hori_lat'], 'g')
    #     ax1.set_title("ankle")
    #     ax2.fill_betweenx(y=[-2000, 5000], x1=64 * 10, facecolor='black', alpha=0.5)
    #     ax2.plot(x, fog1_data['thigh_vertical'], 'r')
    #     ax2.plot(x, fog1_data['thigh_hori_fw'], 'b')
    #     ax2.plot(x, fog1_data['thigh_hori_lat'], 'g')
    #     ax2.set_title("thigh")
    #     ax3.fill_betweenx(y=[-2000, 5000], x1=64 * 10, facecolor='black', alpha=0.5)
    #     ax3.plot(x, fog1_data['trunk_vertical'], 'r')
    #     ax3.plot(x, fog1_data['trunk_hori_fw'], 'b')
    #     ax3.plot(x, fog1_data['trunk_hori_lat'], 'g')
    #     ax3.set_title("trunk")
    #     plt.show()

    def set_file_name(self, file_name):
        self.file_name = file_name

