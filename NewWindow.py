######################################## 导入库 ################################################
import tkinter as tk
from tkinter import ttk
import cv2, os
import csv  ###处理csv文件
import datetime
import time


######################################### 函数 ################################################
def TakeImages():
    pass

def psw():
    pass

def TrackImages():
    pass


def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200, tick)


######################################## 用户界面 ############################################

global key
key = ''

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
day, month, year = date.split("-")

mont = {'01': 'January',
        '02': 'February',
        '03': 'March',
        '04': 'April',
        '05': 'May',
        '06': 'June',
        '07': 'July',
        '08': 'August',
        '09': 'September',
        '10': 'October',
        '11': 'November',
        '12': 'December'
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

message1 = tk.Label(frame2, text="1)人脸录入  >>>  2)信息保存",bg="pink" ,fg="black" , width=39, height=1,
                    activebackground="yellow", font=('times', 15, ' bold '))
message1.place(x=7, y=260)

message = tk.Label(frame2, text="", bg="LightPink" ,fg="black", width=39, height=1, activebackground="yellow",
                   font=('times', 16, ' bold '))
message.place(x=7, y=450)

lbl3 = tk.Label(frame1, text="考勤", width=20, fg="black", bg="pink", height=1, font=('times', 17, ' bold '))
lbl3.place(x=100, y=115)

res = 0
exists = os.path.isfile("StudentDetails\StudentDetails.csv")
if exists:
    with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
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
takeImg.place(x=30, y=320)
trainImg = tk.Button(frame2, text="信息保存", command=psw, fg="black"  ,bg="HotPink", width=34, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
trainImg.place(x=30, y=380)
trackImg = tk.Button(frame1, text="点击考勤", command=TrackImages,fg="black"  ,bg="HotPink", width=35, height=1,
                     activebackground="white", font=('times', 15, ' bold '))
trackImg.place(x=30, y=50)
quitWindow = tk.Button(frame1, text="退出", command=window.destroy, fg="black"  ,bg="HotPink", width=35, height=1,
                       activebackground="white", font=('times', 15, ' bold '))
quitWindow.place(x=30, y=450)

######################################### 显示界面 ###################################################

window.mainloop()

####################################################################################################