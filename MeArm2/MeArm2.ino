#include <Servo.h>
#include <math.h>

Servo base, fArm, rArm, claw;

/**********************
使用MEGA328p的 nano单片机
PWM8、9、10、11作为PWM输出
PWM端口----拓展板上线的标号
6----------6---最顶端舵机--c
7----------7----最底部旋转舵机f
8----------8---上下角度调节，正对机械臂，左侧的舵机--r
9----------9-----前后伸缩，正对机械臂，右侧的舵机--b
因此需要进行调换6-9，9-6
需要注意，最底部舵机姿态初始化调整存在问题，需要等待姿态调整完毕，再发送命令

***********************/

// 存储电机极限值(const指定该数值为常量,常量数值在程序运行中不能改变)
const int baseMin = 0;  // 前后伸缩调节,调节范围较小，仔细控制z
const int baseMax = 180;
const int rArmMin = 0;   // 上下角度调节0-180
const int rArmMax = 180;
const int fArmMin = 30;  // 最底部旋转舵机调节出，以90为中心，
const int fArmMax = 150;
const int clawMin = 0;
const int clawMax = 180; // 最顶部舵机调节

//期盼坐标数据
const float row[] = {-3.3,0.2,3.2,-3.3,0.4,3.5,-3.4,0.2,3.5};//X轴
const float cul[] = {16.3,16.2,16,20.2,20,19.8,24,23.8,23.7};//Y轴

const float Black_X[] = {-6.8,-6.8,-7.0,-7.4,-7.4};//机械臂左侧黑子的X坐标
const float Black_Y[] = {13.2,16.6,20.2,24,27.6};//Y坐标

const float White_X[] = {7.2,7.8,7.8,7.8,7.8};//机械臂右侧白子的X坐标
const float White_Y[] = {12,15.6,19.6,23.4,26.8};//Y坐标

int DSD = 15; // Default Servo Delay (默认电机运动延迟时间)

void setup() {
  // 舵机接口初始化
  base.attach(9);     // base 伺服舵机连接引脚11 舵机代号'b'
  delay(200);         // 稳定性等待
  rArm.attach(8);     // rArm 伺服舵机连接引脚10 舵机代号'r'
  delay(200);         // 稳定性等待
  fArm.attach(7);     // fArm 伺服舵机连接引脚9  舵机代号'f'
  delay(200);         // 稳定性等待
  claw.attach(6);     // claw 伺服舵机连接引脚6  舵机代号'c'
  delay(200);         // 稳定性等待

  // 舵机姿态 初始化
  base.write(180  );   // 前后伸缩调节，初始值设置为160
  delay(20);
  fArm.write(90);    // 最底部旋转舵机f,初始值设置为90
  delay(20);
  rArm.write(50);    // 上下角度调节，初始值设置为50
  delay(20);
  claw.write(90);   // 最顶端舵机c,初始值设置为90
  delay(20);

  // 气泵模块
  pinMode(10, OUTPUT);
  pinMode(11, OUTPUT);
  digitalWrite(10, LOW);
  digitalWrite(11, LOW);

  // 串口模块初始化
  Serial.begin(9600);
  Serial.println("Serial_test");
}

/*
 * 驱动下棋：下来取棋子的时候，高度2-1-0
 * 正常高度是4cm
 * 放下棋子只需要在h=2cm放下
  */

void loop() {
  if (Serial.available() > 0) {
    char serialCmd = Serial.read();
    armDataCmd(serialCmd);
  }
}

void armDataCmd(char serialCmd) {
//  float X ,Y ,Z;
//  float j1, j2, j3;
  if (serialCmd == 'b' || serialCmd == 'c' || serialCmd == 'f' || serialCmd == 'r') {
    int servoData = Serial.parseInt();
    servoCmd(serialCmd, servoData, DSD); // 机械臂舵机运行函数（参数：舵机名，目标角度，延迟/速度）
  } 
  //*************串口首先x之后坐标调试机械臂***********
  else if (serialCmd == 'x') { // 新增指令'x'用于接收坐标
    float X = Serial.parseFloat();
    float Y = Serial.parseFloat();
    float Z = Serial.parseFloat();
    float j1, j2, j3;
    Serial.print("receive x ");
    Serial.print((int)(X));
    Serial.print(" Y ");
    Serial.print((int)(Y));
    Serial.print(" Z ");
    Serial.println((int)(Z));
    if(Z != 2)//Z轴不是2cm
    {
      calculateServoAngles(X, Y, Z, j1, j2, j3); // 调用解算函数计算舵机角度
      Serial.print("cal theta1 ");
      Serial.print(j1);
      Serial.print(" theta2 ");
      Serial.print(j2);
      Serial.print(" theta3 ");
      Serial.println(j3);
      // 发送计算后的角度给舵机
      servoCmd('f', j1, DSD); // 底座角
      servoCmd('b', j2, DSD); // 大臂角
      servoCmd('r', j3, DSD); // 小臂角
    }
    else//Z轴是2cm，逐渐下降
    {
      for(float i = 0;i<2.5;i+=0.5)
      {
          // 调用解算函数计算舵机角度，分阶段下降
          calculateServoAngles(X, Y, Z-i, j1, j2, j3); // 调用解算函数计算舵机角度
          Serial.print("cal theta1 ");
          Serial.print(j1);
          Serial.print(" theta2 ");
          Serial.print(j2);
          Serial.print(" theta3 ");
          Serial.println(j3);
          // 发送计算后的角度给舵机
          servoCmd('f', j1, DSD); // 底座角
          servoCmd('b', j2, DSD); // 大臂角
          servoCmd('r', j3, DSD); // 小臂角
      }
    }
  }
  else if(serialCmd == 'k')//气泵打开
  {
      Serial.print(" Start吸气! ");
      digitalWrite(10, LOW);
      digitalWrite(11, HIGH);
  }
  else if(serialCmd == 'g')//气泵关闭
  {
      Serial.print(" Stop放气! ");
      digitalWrite(10, HIGH );
      digitalWrite(11, LOW);
  }
  //***********根据标号吸收棋盘上的棋子*************
  else if(serialCmd == 'A')//移动坐标,输入对应点的标号就可以直接到达
  {
    //第一步读取坐标，
    int Dot = Serial.parseFloat();
    float X,Y,Z;
    float j1, j2, j3;
    Dot = Dot - 1;
    X = row[Dot];//行，取余
    Y = cul[Dot];//列，整除
    Z = 5;
    digitalWrite(10, HIGH);//开灯
    
    //第二步机械臂移动特定坐标(X,Y,5)
    calculateServoAngles(X, Y, Z, j1, j2, j3); // 调用解算函数计算舵机角度
    Serial.print("cal theta1 ");
    Serial.print(j1);
    Serial.print(" theta2 ");
    Serial.print(j2);
    Serial.print(" theta3 ");
    Serial.println(j3);
    // 发送计算后的角度给舵机
    servoCmd('f', j1, DSD); // 底座角
    servoCmd('b', j2, DSD); // 大臂角
    servoCmd('r', j3, DSD); // 小臂角
    
    //第四步下降Z轴坐标到3cm
    calculateServoAngles(X, Y, 3, j1, j2, j3); // XY不发生变化，Z先调整到3cm
    servoCmd('f', j1, DSD); // 底座角
    servoCmd('b', j2, DSD); // 大臂角
    servoCmd('r', j3, DSD); // 小臂角
    
    //第五步逐渐下降机械臂，吸收棋子
    for(float i = 0;i<3;i+=0.5)
    {
        // 调用解算函数计算舵机角度，分阶段下降
        calculateServoAngles(X, Y, 2-i, j1, j2, j3); // 调用解算函数计算舵机角度
        servoCmd('f', j1, DSD); // 底座角
        servoCmd('b', j2, DSD); // 大臂角
        servoCmd('r', j3, DSD); // 小臂角
    }

    //第三步打开气泵
    digitalWrite(11, HIGH);
    
    //第六步机械臂抬起到5cm，等待放置命令
    calculateServoAngles(X, Y, 5, j1, j2, j3); // 调用解算函数计算舵机角度
    servoCmd('f', j1, DSD); // 底座角
    servoCmd('b', j2, DSD); // 大臂角
    servoCmd('r', j3, DSD); // 小臂角
  }

  //***********根据标号吸收棋盘边上的的棋子——黑子*************
  else if(serialCmd == 'C')//移动坐标,输入对应点的标号就可以直接到达
  {
    //第一步读取坐标，
    int Dot = Serial.parseFloat();
    float X,Y,Z;
    float j1, j2, j3;
    Dot = Dot - 1;
    X = Black_X[Dot];//行，取余
    Y = Black_Y[Dot];//列，整除
    Z = 5;
    digitalWrite(10, HIGH);//开灯
    
    //第二步机械臂移动特定坐标(X,Y,5)
    calculateServoAngles(X, Y, Z, j1, j2, j3); // 调用解算函数计算舵机角度
    Serial.print("receive x ");
    Serial.print((int)(X));
    Serial.print(" Y ");
    Serial.print((int)(Y));
    Serial.print(" Z ");
    Serial.println((int)(Z));
    Serial.print("cal theta1 ");
    Serial.print(j1);
    Serial.print(" theta2 ");
    Serial.print(j2);
    Serial.print(" theta3 ");
    Serial.println(j3);
    // 发送计算后的角度给舵机
    servoCmd('f', j1, DSD); // 底座角
    servoCmd('b', j2, DSD); // 大臂角
    servoCmd('r', j3, DSD); // 小臂角
    
    //第四步下降Z轴坐标到3cm
    calculateServoAngles(X, Y, 3, j1, j2, j3); // XY不发生变化，Z先调整到3cm
    servoCmd('f', j1, DSD); // 底座角
    servoCmd('b', j2, DSD); // 大臂角
    servoCmd('r', j3, DSD); // 小臂角
    
    //第五步逐渐下降机械臂，吸收棋子
    for(float i = 0;i<3;i+=0.5)
    {
        // 调用解算函数计算舵机角度，分阶段下降
        calculateServoAngles(X, Y, 2-i, j1, j2, j3); // 调用解算函数计算舵机角度
        servoCmd('f', j1, DSD); // 底座角
        servoCmd('b', j2, DSD); // 大臂角
        servoCmd('r', j3, DSD); // 小臂角
    }
        
    //第三步打开气泵
    
    digitalWrite(11, HIGH);
    
    //第六步机械臂抬起到5cm，等待放置命令
    calculateServoAngles(X, Y, 5, j1, j2, j3); // 调用解算函数计算舵机角度
    servoCmd('f', j1, DSD); // 底座角
    servoCmd('b', j2, DSD); // 大臂角
    servoCmd('r', j3, DSD); // 小臂角
  }
  //***********根据标号吸收棋盘边上的的棋子——白子*************
  else if(serialCmd == 'D')//移动坐标,输入对应点的标号就可以直接到达
  {
    //第一步读取坐标，
    int Dot = Serial.parseFloat();
    float X,Y,Z;
    float j1, j2, j3;
    Dot = Dot - 1;
    X = White_X[Dot];//行，取余
    Y = White_Y[Dot];//列，整除
    Z = 5;
    digitalWrite(10, HIGH);//开灯
    
    //第二步机械臂移动特定坐标(X,Y,5)
    calculateServoAngles(X, Y, Z, j1, j2, j3); // 调用解算函数计算舵机角度
    Serial.print("cal theta1 ");
    Serial.print(j1);
    Serial.print(" theta2 ");
    Serial.print(j2);
    Serial.print(" theta3 ");
    Serial.println(j3);
    // 发送计算后的角度给舵机
    servoCmd('f', j1, DSD); // 底座角
    servoCmd('b', j2, DSD); // 大臂角
    servoCmd('r', j3, DSD); // 小臂角
    
    
    //第四步下降Z轴坐标到3cm
    calculateServoAngles(X, Y, 3, j1, j2, j3); // XY不发生变化，Z先调整到3cm
    servoCmd('f', j1, DSD); // 底座角
    servoCmd('b', j2, DSD); // 大臂角
    servoCmd('r', j3, DSD); // 小臂角
    
    //第五步逐渐下降机械臂，吸收棋子
    for(float i = 0;i<3;i+=0.5)
    {
        // 调用解算函数计算舵机角度，分阶段下降
        calculateServoAngles(X, Y, 2-i, j1, j2, j3); // 调用解算函数计算舵机角度
        servoCmd('f', j1, DSD); // 底座角
        servoCmd('b', j2, DSD); // 大臂角
        servoCmd('r', j3, DSD); // 小臂角
    }
    //第三步打开气泵
    //digitalWrite(10, HIGH);//开灯
    digitalWrite(11, HIGH);
    
    //第六步机械臂抬起到5cm，等待放置命令
    calculateServoAngles(X, Y, 5, j1, j2, j3); // 调用解算函数计算舵机角度
    servoCmd('f', j1, DSD); // 底座角
    servoCmd('b', j2, DSD); // 大臂角
    servoCmd('r', j3, DSD); // 小臂角
  }
  //****************根据标号放置棋子*******************
  else if(serialCmd == 'B')
  {
    //第一步读取坐标，
    int Dot = Serial.parseFloat();
    float X,Y,Z;
    float j1, j2, j3;
    Dot = Dot - 1;
    X = row[Dot];//行，取余
    Y = cul[Dot];//列，整除
    Z = 5;
    //第二步机械臂移动特定坐标(X,Y,5)
    calculateServoAngles(X, Y, Z, j1, j2, j3); // 调用解算函数计算舵机角度
    Serial.print("cal theta1 ");
    Serial.print(j1);
    Serial.print(" theta2 ");
    Serial.print(j2);
    Serial.print(" theta3 ");
    Serial.println(j3);
    // 发送计算后的角度给舵机
    servoCmd('f', j1, DSD); // 底座角
    servoCmd('b', j2, DSD); // 大臂角
    servoCmd('r', j3, DSD); // 小臂角
    
    //第三步下降到1cm
    calculateServoAngles(X, Y, 1, j1, j2, j3); // XY不发生变化，Z先调整到3cm
    servoCmd('f', j1, DSD); // 底座角
    servoCmd('b', j2, DSD); // 大臂角
    servoCmd('r', j3, DSD); // 小臂角
    delay(500);
    //第四步直接放气落子
    digitalWrite(10, LOW);//提示灯
    digitalWrite(11, LOW);

    delay(4000);

    //第五步机械臂归位
    calculateServoAngles(X, Y, 3, j1, j2, j3); // XY不发生变化，Z先调整到3cm
    servoCmd('f', j1, DSD); // 底座角
    servoCmd('b', j2, DSD); // 大臂角
    servoCmd('r', j3, DSD); // 小臂角
    armIniPos();
  }
  //************输出信息和舵机归位***************
  else {
    switch (serialCmd) {
      case 'o': // 输出舵机状态信息
        reportStatus();
        break;
      case 'i': // 舵机姿态归位
        armIniPos();
        break;
      default: // 未知指令反馈
        Serial.println("Unknown Command.");
    }
  }
}

//舵机命令函数
void servoCmd(char servoName, int toPos, int servoDelay) {
  Servo servo2go;

  Serial.println("");
  Serial.print("+Command: Servo ");
  Serial.print(servoName);
  Serial.print(" to ");
  Serial.print(toPos);
  Serial.print(" at servoDelay value ");
  Serial.print(servoDelay);
  Serial.println(".");
  Serial.println("");

  int fromPos;

  switch (servoName) {
    case 'b':
      if (toPos >= baseMin && toPos <= baseMax) {
        servo2go = base;
        fromPos = base.read();
        break;
      } else {
        Serial.println("+Warning: Base Servo Value Out Of Limit!");
        return;
      }
    case 'c':
      if (toPos >= clawMin && toPos <= clawMax) {
        servo2go = claw;
        fromPos = claw.read();
        break;
      } else {
        Serial.println("+Warning: Claw Servo Value Out Of Limit!");
        return;
      }
    case 'f':
      if (toPos >= fArmMin && toPos <= fArmMax) {
        servo2go = fArm;
        fromPos = fArm.read();
        break;
      } else {
        Serial.println("+Warning: fArm Servo Value Out Of Limit!");
        return;
      }
    case 'r':
      if (toPos >= rArmMin && toPos <= rArmMax) {
        servo2go = rArm;
        fromPos = rArm.read();
        break;
      } else {
        Serial.println("+Warning: rArm Servo Value Out Of Limit!");
        return;
      }
  }

  if (fromPos <= toPos) {
    for (int i = fromPos; i <= toPos; i++) {
      servo2go.write(i);
      delay(servoDelay);
    }
  } else {
    for (int i = fromPos; i >= toPos; i--) {
      servo2go.write(i);
      delay(servoDelay);
    }
  }
}

void reportStatus() {
  Serial.println("");
  Serial.println("+ Robot-Arm Status Report +");
  Serial.print("Claw Position: "); Serial.println(claw.read());
  Serial.print("Base Position: "); Serial.println(base.read());
  Serial.print("Rear Arm Position: "); Serial.println(rArm.read());
  Serial.print("Front Arm Position: "); Serial.println(fArm.read());
  Serial.println("++++++++++++++++++++++++++");
  Serial.println("");
}

void armIniPos() {
  Serial.println("+Command: Restore Initial Position.");
  int robotIniPosArray[4][3] = {
    {'b', 160, DSD},
    {'r', 50, DSD},
    {'f', 90, DSD},
    {'c', 180, DSD}
  };

  for (int i = 0; i < 4; i++) {
    servoCmd(robotIniPosArray[i][0], robotIniPosArray[i][1], robotIniPosArray[i][2]);
  }
}

void calculateServoAngles(float X, float Y, float Z, float& j1, float& j2, float& j3) {
  // 定义机械臂的长度和补偿量
  float baseHeight = 4.5;      // 底座高度
  float armLength = 14;        // 大臂长度
  float forearmLength = 14.5;  // 小臂长度
  float armCompensation = 3;   // 臂向补偿
  float heightCompensation = -3; // 高度补偿
  Z = Z - heightCompensation;
  // 计算底座角度
  j1 = atan2(Y, X) * 57.2958; // 转换为度数, atan2 是 Y/X，然后和 X 轴正方向夹角

  // 计算大臂角度
  float verticalProjection = sqrt(X * X + Y * Y) - armCompensation; // 垂直投影
  float shortSide = fabs(baseHeight - Z); // 短边
  float hypotenuse = sqrt(shortSide * shortSide + verticalProjection * verticalProjection); // 斜边

  // 大臂角度计算
  float armAngle1 = acos((armLength * armLength + hypotenuse * hypotenuse - forearmLength * forearmLength) / (2 * armLength * hypotenuse)) * 57.2958; // 转换为度数:acos反余弦为0-pi的弧度
  float armAngle2 = 0;
  if (baseHeight == Z) {//高度相等
    armAngle2 = 0;
  } else if (baseHeight > Z) {
    armAngle2 = atan2(verticalProjection, shortSide) * 57.2958 - 90; // 转换为度数，因为 90 度为舵机 0 度，所以 -90
  } else if (baseHeight < Z) {
    armAngle2 = atan2(shortSide, verticalProjection) * 57.2958 ; // 转换为度数
  }
  j2 = armAngle1 + armAngle2 ; // 大臂角度总和

  // 小臂角度计算
  //180-
  j3 = 180-(acos((armLength * armLength + forearmLength * forearmLength - hypotenuse * hypotenuse) / (2 * armLength * forearmLength)) * 57.2958 + j2); // 转换为度数

  // 限制角度范围
  j2 = 1.5*j2;
  j3 = 1.5*j3;//补偿1.5倍

  if (j1 < fArmMin) j1 = fArmMin;  // 底座 0-180
  if (j1 > fArmMax) j1 = fArmMax;
  if (j2 < baseMin) j2 = baseMin;  // 大臂 0-180
  if (j2 > baseMax) j2 = baseMax;
  if (j3 < rArmMin) j3 = rArmMin;  // 小臂 0-180
  if (j3 > rArmMax) j3 = rArmMax;
}
