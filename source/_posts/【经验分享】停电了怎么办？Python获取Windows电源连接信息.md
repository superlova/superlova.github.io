---
title: 【经验分享】停电了怎么办？Python获取Windows电源连接信息
date: 2020-06-08 09:42:45
index_img: /img/poweroff.jpg
tags: ['Windows', 'Python', 'PowerOff']
categories: 
- record
---
一旦停电，就令笔记本电脑发出响声、发送信息……。看似简单的功能，该如何利用Python实现呢？
<!--more--->

采用笔记本电脑办公的好处是不必害怕突然停电。然而笔记本电脑不可能使用电池工作太久，此时必须尽快通知管理人员，恢复供电。

看似简单的功能，只需在Windows中注册一个HANDLE，负责接收电源适配器更改这一事件即可。但是本人没有Windows编程和系统编程的经验，只对Python熟悉。如何实现这一功能？

废话不多说，下面是代码。

```python
import win32con
import win32api
import win32gui
import time
from ctypes import POINTER, windll, Structure, cast, CFUNCTYPE, c_int, c_uint, c_void_p, c_bool
from comtypes import GUID
from ctypes.wintypes import HANDLE, DWORD

PBT_POWERSETTINGCHANGE = 0x8013
GUID_CONSOLE_DISPLAY_STATE = '{6FE69556-704A-47A0-8F24-C28D936FDA47}'
GUID_ACDC_POWER_SOURCE = '{5D3E9A59-E9D5-4B00-A6BD-FF34FF516548}'
GUID_BATTERY_PERCENTAGE_REMAINING = '{A7AD8041-B45A-4CAE-87A3-EECBB468A9E1}'
GUID_MONITOR_POWER_ON = '{02731015-4510-4526-99E6-E5A17EBD1AEA}'
GUID_SYSTEM_AWAYMODE = '{98A7F580-01F7-48AA-9C0F-44352C29E5C0}'

class POWERBROADCAST_SETTING(Structure):
    _fields_ = [("PowerSetting", GUID),
                ("DataLength", DWORD),
                ("Data", DWORD)]


def wndproc(hwnd, msg, wparam, lparam):
    if msg == win32con.WM_POWERBROADCAST:
        if wparam == win32con.PBT_APMPOWERSTATUSCHANGE:
            print('Power status has changed')
        if wparam == win32con.PBT_APMRESUMEAUTOMATIC:
            print('System resume')
        if wparam == win32con.PBT_APMRESUMESUSPEND:
            print('System resume by user input')
        if wparam == win32con.PBT_APMSUSPEND:
            print('System suspend')
        if wparam == PBT_POWERSETTINGCHANGE:
            print('Power setting changed...')
            settings = cast(lparam, POINTER(POWERBROADCAST_SETTING)).contents
            power_setting = str(settings.PowerSetting)
            data_length = settings.DataLength
            data = settings.Data
            if power_setting == GUID_CONSOLE_DISPLAY_STATE:
                if data == 0: print('Display off')
                if data == 1: print('Display on')
                if data == 2: print('Display dimmed')
            elif power_setting == GUID_ACDC_POWER_SOURCE:
                if data == 0: print('AC power')
                if data == 1:
                    print('Battery power')
                    #################################################
                    playsound('alert.mp3') # 此处自定义你的操作
                    #################################################
                if data == 2: print('Short term power')
            elif power_setting == GUID_BATTERY_PERCENTAGE_REMAINING:
                print('battery remaining: %s' % data)
            elif power_setting == GUID_MONITOR_POWER_ON:
                if data == 0: print('Monitor off')
                if data == 1: print('Monitor on')
            elif power_setting == GUID_SYSTEM_AWAYMODE:
                if data == 0: print('Exiting away mode')
                if data == 1: print('Entering away mode')
            else:
                print('unknown GUID')
        return True

    return False




if __name__ == "__main__":
    print("*** STARTING ***")
    hinst = win32api.GetModuleHandle(None)
    wndclass = win32gui.WNDCLASS()
    wndclass.hInstance = hinst
    wndclass.lpszClassName = "testWindowClass"
    CMPFUNC = CFUNCTYPE(c_bool, c_int, c_uint, c_uint, c_void_p)
    wndproc_pointer = CMPFUNC(wndproc)
    wndclass.lpfnWndProc = {win32con.WM_POWERBROADCAST : wndproc_pointer}
    try:
        myWindowClass = win32gui.RegisterClass(wndclass)
        hwnd = win32gui.CreateWindowEx(win32con.WS_EX_LEFT,
                                     myWindowClass,
                                     "testMsgWindow",
                                     0,
                                     0,
                                     0,
                                     win32con.CW_USEDEFAULT,
                                     win32con.CW_USEDEFAULT,
                                     0,
                                     0,
                                     hinst,
                                     None)
    except Exception as e:
        print("Exception: %s" % str(e))

    if hwnd is None:
        print("hwnd is none!")
    else:
        print("hwnd: %s" % hwnd)

    guids_info = {
                    'GUID_MONITOR_POWER_ON' : GUID_MONITOR_POWER_ON,
                    'GUID_SYSTEM_AWAYMODE' : GUID_SYSTEM_AWAYMODE,
                    'GUID_CONSOLE_DISPLAY_STATE' : GUID_CONSOLE_DISPLAY_STATE,
                    'GUID_ACDC_POWER_SOURCE' : GUID_ACDC_POWER_SOURCE,
                    'GUID_BATTERY_PERCENTAGE_REMAINING' : GUID_BATTERY_PERCENTAGE_REMAINING
                 }
    for name, guid_info in guids_info.items():
        result = windll.user32.RegisterPowerSettingNotification(HANDLE(hwnd), GUID(guid_info), DWORD(0))
        print('registering', name)
        print('result:', hex(result))
        print('lastError:', win32api.GetLastError())
        print()

    print('\nEntering loop')
    while True:
        win32gui.PumpWaitingMessages()
        time.sleep(1)
```

COM: The Component Object Model 组件对象模型，是微软的一套软件组件的二进制接口标准。COM使得跨编程语言的进程间通信、动态对象创建成为可能。

COM实质上是一种语言无关的对象实现方式，这使其可以在创建环境不同的场合、甚至跨计算机的分布环境下被复用。COM允许复用这些对象，而不必知道对象内部是如何实现，因为组件实现者必须提供良好定义的接口从而屏蔽实现细节。通过引用计数，组件对象自己负责动态创建与销毁，从而屏蔽了不同编程语言之间的内存分配语义差异。

对于某些应用程序来说，COM已经部分被.NET框架取代。.NET Framework是新一代的Microsoft Windows应用程序开发平台。

COM是基于组件对象方式概念来设计的，在基础中，至少要让每个组件都可以支持二个功能：

查询组件中有哪些接口
让组件做自我生命管理，此概念的实践即为引用计数（Reference Counting）

GUID 是一个 128 位整数（16 字节），COM将其用于计算机和网络的唯一标识符。全局唯一标识符（英语：Globally Unique Identifier，缩写：GUID；发音为/ˈɡuːɪd/或/ˈɡwɪd/）是一种由算法生成的唯一标识，通常表示成32个16进制数字（0－9，A－F）组成的字符串，如：{21EC2020-3AEA-1069-A2DD-08002B30309D}，它实质上是一个128位长的二进制整数。

Windows操作系统使用GUID来标识COM对象中的类和界面。一个脚本可以不需知道DLL的位置和名字直接通过GUID来激活其中的类或对象。

参考：https://stackoverflow.com/questions/48720924/python-3-detect-monitor-power-state-in-windows