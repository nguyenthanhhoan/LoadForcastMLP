import csv, re, os
import pandas as pd
import shutil

#Dùng để  lấy dữ liệu từ file csv và lưu nó vào list
class CSVDataPreProcessor():
    def __init__(self,file_name,feature_size = 100,label_size = 5):
        """
        Khởi tạo biến, với:
        file_name_prefix_input = "dbArchive"
        feature_size = 100 -> số lượng đầu vào
        label_size = 5 -> Số lượng đầu ra
        """
        self.file_name = file_name
        #CHECK IF FILE NAME IS EXISTS
        try:
            dump = open(self.file_name)
        except FileNotFoundError:
            print("Your file is not found! Please check your path and try again")
            exit(-1)
        dump.close()
        self.feature_size = feature_size
        self.label_size = label_size

    def get_file_name(self):
        return self.file_name

    def get_feature_size(self):
        return self.feature_size
    
    def get_label_size(self):
        return self.feature_size

    def get_directory(self,column_name):
        directory = "./" + self.file_name[:len(self.file_name) - 4] + "/" + column_name
        return directory


    def load_column_data(self,column_name):
        directory = "./" + self.file_name[:len(self.file_name) - 4] + "/" + column_name
        """
        Từ tập tin data, lấy dữ liệu dựa trên tên cột, rồi tách dữ liệu đó ra thành 2 loại: train và test
        """
        #Tạo thư mục cho tập tin đó
        try:
            os.mkdir(self.file_name[:len(self.file_name) - 4])
        except:
            print("Warning: Folder of {} has been created. Overwriting them...".format(self.file_name))
        try:
            os.mkdir(self.file_name[:len(self.file_name) - 4] + "/" + column_name)
        except:
            print("Warning: Column {} of {} has been created. Overwriting them...".format(column_name,self.file_name))
        #Trong thư mục vừa mới tạo ở trên, tách các dữ liệu các cột thành các thư mục riêng biệt
        train_data = None
        label_data = None
        if os.path.exists(directory + "/train.csv"):
            print("Warning:{}: Its train data has been created. Overwriting them...".format(self.file_name))
            os.remove(directory + "/train.csv")
        train_data = open(directory + "/train.csv",'a')

        if os.path.exists(directory + "/label.csv"):
            print("Warning:{}: Its label data has been created. Overwriting them...".format(self.file_name))
            os.remove(directory + "/label.csv")
        label_data = open(directory + "/label.csv",'a')

        #Đọc tập tin từ file_name
        df = pd.read_csv(self.file_name,sep=',',header=(0))
        #Tách dữ liệu cột x thành 2 loại: train và test
        for t in range(0,len(df[column_name]) - self.feature_size - self.label_size):
            train_lines = df[column_name][t : t + self.feature_size].values
            label_lines = df[column_name][t + self.feature_size : t + self.feature_size + self.label_size].values
            train_lines = ','.join(str(v) for v in train_lines)
            label_lines = ','.join(str(v) for v in label_lines)
            train_data.write(train_lines + "\n")
            label_data.write(label_lines + "\n")
        
        #Đóng tập tin
        train_data.close()
        label_data.close()
        """
        Từ tên tập tin, lấy dữ liệu và lưu nó vào biến dưới 2 loại: dataX (thông tin data), và dataY (kết quả tương ứng của data đó).
        """
        with open(directory + "/train.csv", 'r') as train_data:
            dataX = train_data.readlines()
        with open(directory + "/label.csv", 'r') as label_data:
            dataY = label_data.readlines()
        def split_and_int(line):
            line = re.split(',',line)
            for i in range(len(line)):
                line[i] = float(line[i])
            return line
        #removing endline character
        dataX = [line[:-1] for line in dataX]
        dataY = [line[:-1] for line in dataY]
        #split string and convert to int
        dataX = [split_and_int(line) for line in dataX]
        dataY = [split_and_int(line) for line in dataY]
        return dataX, dataY
