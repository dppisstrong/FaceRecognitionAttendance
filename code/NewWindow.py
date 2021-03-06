'''
Created on 2022年6月17日
@author: LiJing、DingPan
@description: 本程序为人脸考勤系统界面
@version: 1.6
@copyright: cqut11903991020、cqut11903991019
'''
############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess  ###消息对话框
import os
import csv  ###处理csv文件
import datetime
import time
import pandas as pd

from face_normalize import FaceNormalize
from get_face_img import GetFaceImg
from assure_path import assure_path_exists
from create_dataset import Dataset
from model import Model
from face_recognition import Recognition

######################################### 函数 ################################################
def TakeImages():
    global Id
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME', '', 'Tele', '', 'Sex']  #####
    assure_path_exists("../data/StudentDetails/")
    serial = 0
    exists = os.path.isfile("..\data\StudentDetails\StudentDetails.csv")
    if exists:
        with open("..\data\StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    else:
        with open("..\data\StudentDetails\StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()
    Id = (txt.get())
    name = (txt1.get())
    Tele = (txt2.get())
    Sex = (var.get())
    if ((name.isalpha()) or (' ' in name)):
        Face_Img = GetFaceImg(Id)
        Face_Img.run()
        res = "用户\"" + Id + "\"完成信息录入！"
        row = [serial, '', Id, '', name, '', Tele, '', Sex]
        with open('..\data\StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=res)
    else:
        if (name.isalpha() == False):
            res = "Enter correct name"
            message.configure(text=res)  ####提示信息

def ImgModel():
    global Id
    # 图像归一化
    face_normalize = FaceNormalize()
    face_normalize.run(Id)
    #加载数据集
    dataset = Dataset()
    dataset.load_dataset()
    dataset.prepare_dataset()
    model = Model()
    model.build_model(dataset)
    model.train_model(dataset)
    model.evaluate_model(dataset)
    assure_path_exists('../data/model/')
    model.save_model('../data/model/model.h5')

#考勤
def attendance():
    assure_path_exists("../data/Attendance/")
    assure_path_exists("../data/StudentDetails/")
    # 清空界面原有考勤数据
    for k in tv.get_children():
        tv.delete(k)
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']  # 考勤表列名
    exists1 = os.path.isfile("..\data\StudentDetails\StudentDetails.csv")
    if exists1:
        df = pd.read_csv("..\data\StudentDetails\StudentDetails.csv")
        recognition = Recognition()
        recognition.run()  # 运行人脸识别程序
        if not recognition.get_name_flag:  # 成功识别出人脸
            i = 1
            recognized_name = [int(recognition.my_name)]  # 将人员ID提取出来
            while i <= res:  # 对表中数据依次比对
                if recognized_name == df.loc[df['SERIAL NO.'] == i]['ID'].values:
                    serial = i  # 人脸识别结果的ID对应的学生在表中序号
                    break
                else:
                    i = i + 1
            ts = time.time()  # 获得当前时间
            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')  # 年月日
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')  # 时分秒
            aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
            ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
            ID = str(ID)
            ID = ID[1:-1]
            bb = str(aa)
            bb = bb[2:-2]
            attendance = [str(ID), '', bb, '', str(date), '', str(timeStamp)]
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        exists = os.path.isfile("..\data\Attendance\Attendance_" + date + ".csv")
        if exists:
            # 存在当日考勤数据
            with open("..\data\Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
                writer = csv.writer(csvFile1)
                writer.writerow(attendance)
            csvFile1.close()
        else:
            # 不存在当日考勤数据
            with open("..\data\Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
                writer = csv.writer(csvFile1)
                writer.writerow(col_names)  # 先加入列名
                writer.writerow(attendance)
            csvFile1.close()
        j = 0
        space1 = ''
        space2 = '       '
        space3 = '         '
        with open("..\data\Attendance\Attendance_" + date + ".csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            # 将当日所有考勤信息输出到界面
            for lines in reader1:
                j = j + 1
                if (j > 1):
                    if (j % 2 != 0):
                        iidd = '  '+ str(lines[0]) + '   '
                        i = 0
                        count = 16 - len(str(lines[2]))
                        while i < count:
                            space1 += ' '
                            i = i + 1
                        tv.insert('', 0, text=iidd, values=(space1+str(lines[2]), str(space2+lines[4]), str(space3+lines[6])))
                        space1 = ''  # 重置
        csvFile1.close()
    else:
        mess._show(title='信息丢失', message='学生信息丢失，请进行检查!')
        window.destroy()

###查询
def select():
    def select_attendance():
        new_year = year.get()
        new_month = month.get()
        new_day = day.get()
        new_name = name.get()
        time = "{}-{}-{}".format(new_day, new_month, new_year)
        exists = os.path.isfile("..\data\Attendance\Attendance_" + time + ".csv")
        if exists:
            j = 0
            space1 = ''
            space2 = '       '
            space3 = '         '
            # 清空界面原有考勤数据
            for k in tv.get_children():
                tv.delete(k)
            with open("..\data\Attendance\Attendance_" + time + ".csv", 'r') as csvFile1:
                reader1 = csv.reader(csvFile1)
                # 将当日所有考勤信息输出到界面
                for lines in reader1:
                    j = j + 1
                    if (j > 1):
                        if (j % 2 != 0):
                            if str(lines[2]) == new_name:
                                iidd = '  ' + str(lines[0]) + '   '
                                i = 0
                                count = 16 - len(str(lines[2]))
                                while i < count:
                                    space1 += ' '
                                    i = i + 1
                                tv.insert('', 0, text=iidd, values=(space1+str(lines[2]), str(space2+lines[4]), str(space3+lines[6])))
                                space1 = ''  # 重置
            csvFile1.close()
            window_select.destroy()
        else:
            mess._show(title='查询失败', message='该考勤时间（或用户名）不存在!')

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)

######################################## 用户界面 ############################################

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day, month, year = date.split("-")

mont = {'01': 'January', '02': 'February', '03': 'March', '04': 'April',
        '05': 'May', '06': 'June', '07': 'July', '08': 'August',
        '09': 'September', '10': 'October', '11': 'November', '12': 'December'
        }

############################################ 界面设置 ################################################

window = tk.Tk()
window.geometry("1280x720")
window.resizable(True, False)
window.title("考勤系统")
window.configure(background='Indigo')  ##背景black

frame2 = tk.Frame(window, bg="pink")  ##左边背景blue
frame2.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

frame1 = tk.Frame(window, bg="pink")  ###右边背景blue
frame1.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

message3 = tk.Label(window, text="人脸识别考勤系统", fg="LightGray", bg="Indigo", width=53, height=1,
                    font=('times', 29, ' bold '))
message3.place(x=10, y=10)  ####title字体为purple,背景为black

frame3 = tk.Frame(window, bg="#c4c6ce")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

datef = tk.Label(frame4, text=day + "-" + mont[month] + "-" + year + "  |  ", fg="silver",bg="Indigo" , width=55,
                 height=1, font=('times', 22, ' bold '))
datef.pack(fill='both', expand=1)  ###日期

clock = tk.Label(frame3, fg="silver",bg="Indigo" , width=55, height=1, font=('times', 22, ' bold '))
clock.pack(fill='both', expand=1)  ###时间
tick()

head2 = tk.Label(frame2, text="                            注册新的信息                              ", fg="black",bg="LightPink", font=('times', 17, ' bold '))
head2.grid(row=0, column=0)

head1 = tk.Label(frame1, text="                            信息已经注册                             ", fg="black",bg="LightPink", font=('times', 17, ' bold '))
head1.place(x=0, y=0)

###账号
lbl = tk.Label(frame2, text="账号", width=5, height=1, fg="black"  ,bg="PaleVioletRed" , font=('times', 17, ' bold '))
lbl.place(x=50, y=55)

txt = tk.Entry(frame2, width=20, fg="black", font=('times', 15, ' bold '))
txt.place(x=150, y=57)

###姓名
lbl1 = tk.Label(frame2, text="姓名", width=5, fg="black"  ,bg="PaleVioletRed" , font=('times', 17, ' bold '))
lbl1.place(x=50, y=100)

txt1 = tk.Entry(frame2, width=20, fg="black", font=('times', 15, ' bold '))
txt1.place(x=150, y=102)
###手机号
lbl2 = tk.Label(frame2, text="手机号", width=5, height=1, fg="black"  ,bg="PaleVioletRed" , font=('times', 17, ' bold '))
lbl2.place(x=50, y=145)

txt2 = tk.Entry(frame2, width=20, fg="black", font=('times', 15, ' bold '))
txt2.place(x=150, y=147)

###性别
var = tk.StringVar()
var.set('man')

lbl0 = tk.Label(frame2, text="性别", width=5, height=1, fg="black"  ,bg="PaleVioletRed" , font=('times', 17, ' bold '))
lbl0.place(x=50, y=190)

r1 = tk.Radiobutton(window, text='男', variable=var, value='man', bg="pink", font=('times', 13, ' bold '))
r1.place(x=330, y=315)
r2 = tk.Radiobutton(window, text='女', variable=var, value='woman', bg="pink", font=('times', 13, ' bold '))
r2.place(x=405, y=315)

message1 = tk.Label(frame2, text="1)人脸录入  >>>  2)创建模型",bg="pink" ,fg="black" , width=39, height=1,
                    activebackground="yellow", font=('times', 15, ' bold '))
message1.place(x=7, y=260)

message = tk.Label(frame2, text="", bg="LightPink" ,fg="black", width=39, height=1, activebackground="yellow",
                   font=('times', 16, ' bold '))
message.place(x=7, y=450)

lbl3 = tk.Label(frame1, text="考勤", width=20, fg="black", bg="pink", height=1, font=('times', 17, ' bold '))
lbl3.place(x=100, y=115)

res = 0
exists = os.path.isfile("..\data\StudentDetails\StudentDetails.csv")
if exists:
    with open("..\data\StudentDetails\StudentDetails.csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for l in reader1:
            res = res + 1
    res = (res // 2) - 1
    csvFile1.close()
else:
    res = 0
message.configure(text='已注册用户人数  : ' + str(res))

############################################ 考勤表 ####################################################

tv = ttk.Treeview(frame1, height=13, columns=('name', 'date', 'time'))
tv.column('#0', width=82)
tv.column('name', width=130)
tv.column('date', width=133)
tv.column('time', width=133)
tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)
tv.heading('#0', text='账号')
tv.heading('name', text='姓名')
tv.heading('date', text='日期')
tv.heading('time', text='时间')

############################################ 滑动条 #####################################################

scroll = ttk.Scrollbar(frame1, orient='vertical', command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky='ns')
tv.configure(yscrollcommand=scroll.set)

############################################ 按钮 #######################################################

takeImg = tk.Button(frame2, text="人脸录入", command=TakeImages,fg="black"  ,bg="HotPink", width=34, height=1,
                    activebackground="white", font=('times', 15, ' bold '))
takeImg.place(x=50, y=320)
Modeling = tk.Button(frame2, text="创建模型", command=ImgModel, fg="black"  ,bg="HotPink", width=34, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
Modeling.place(x=50, y=380)
Att = tk.Button(frame1, text="点击考勤", command=attendance,fg="black"  ,bg="HotPink", width=15, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
Att.place(x=45, y=50)
selection = tk.Button(frame1, text="查询考勤", command=select,fg="black"  ,bg="HotPink", width=15, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
selection.place(x=260, y=50)
quitWindow = tk.Button(frame1, text="退出", command=window.destroy, fg="black"  ,bg="HotPink", width=35, height=1,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=30, y=450)

######################################### 显示界面 ###################################################

window.mainloop()

####################################################################################################
