from data_preprocessor import CSVDataPreProcessor
from MLPLayer import MLP
import sys
input_size = 5 #Lấy 100 số làm input, mỗi kết quả cách nhau 1 quãng thời gian bằng nhau
output_size = 1 #Dự đoán 5 số vol, mỗi số vol cách nhau 1 quãng thời gian bằng nhau
n_hidden = 3
batch_size = 10
learning_rate = 0.05
training_epoch = 100 #fix
num_folds = 8
#Getting system parameters
if len(sys.argv) != 3:
    print("Please type according to this format: python3 main.py [file name] [column name]")
    exit(-1)
file_name = sys.argv[1]
column_name = sys.argv[2]
data = CSVDataPreProcessor(file_name,input_size,output_size) #Khởi tạo biến
X,Y = data.load_column_data(column_name)
MLP_instance = MLP(input_size, n_hidden, output_size,learning_rate,batch_size,training_epoch)
MLP_instance.train_save_and_log(X,Y,num_folds,'./' + file_name[:len(file_name) - 4] + '/' + column_name,file_name[:len(file_name) - 4] + '/' + column_name + '/model') 