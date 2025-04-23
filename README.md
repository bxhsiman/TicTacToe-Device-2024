# 2024年电赛E题-三子棋游戏装置
## 项目结构
``` bash
.
├── README.md
├── MeArm2 
│   ├── MeArm2.ino # 机械臂控制程序
├── refcode # 参考代码库
├── vision # 上位机程序
│   ├── GUI_tk.py # 主程序 GUI
│   ├── program*.py # 任务子程序
├── 报告.pdf # 设计报告
├── E题_三子棋游戏装置.pdf # 本题题目
```
## 设备构成
本项目使用了以下设备:
- 三轴机械臂(带真空吸泵)
- MEGA328P Nano、MEGA2560
- 树莓派zero2w 
- 6寸触摸显示器
## 代码结构
本项目下位机使用arduino框架开发，上位机使用opencv实现视觉识别逻辑，并使用thinker实现gui，结合树莓派自启任务完成控制。上下位机之间采用串口通信。
## 空间解算算法解析
报告写了点，不详细的代码注释也很全面
## 视觉识别算法解析
报告写了点，不详细的代码注释也很全面
## 项目完成度
- 没有实现旋转识别任务
- 本仓库的策略代码不保证下棋策略最优，仅供参考
- 24年北京赛区省一

## 作者信息(Author list)
- siman@qiulishe.com 
- [IamFeynman](https://github.com/IamFeynman)
- 1928857720@qq.com


BUPT-2024 




