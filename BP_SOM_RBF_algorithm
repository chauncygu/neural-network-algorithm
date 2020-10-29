

%%%%采用BP神经网络进行数据拟合
clear all
load simplefit_dataset
figure(1);
plot(simplefitInputs, simplefitTargets,'+');
[x,t]=simplefit_dataset;
net=fitnet(10);
net=train(net,x,t);
view(net);
y=net(x);
perf=perform(net,t,y);
% figure(2);
hold on;
plot(simplefitInputs,y)

%%%采用BP神将网络进行数据分类
clear all
close all
load iris_dataset
[x,t]=iris_dataset;
net=patternnet(10);
net=train(net,x,t);
view(net);


%%%%%%%采用SOM网络进行聚类
clear all
load simplecluster_dataset;
[x,t]=simplecluster_dataset;
plot(x(1,:),x(2,:),'+');
dimension1=10;
dimension2=10;
net=selforgmap([dimension1,dimension2]);
view(net);
net=train(net,x);  
y=net(x);
classes=vec2ind(y);


%%%%采用RBF网络进行数据拟合
clear all
X=-1:0.1:1;
T=[-0.9602,-0.5770,-0.0729,0.3771,0.6405,0.6600,0.4609 ...
    0.1336,-0.2013,-0.4344,-0.5,-0.393,-0.1647,0.0988 ...
    0.3072,0.3960,0.3449,0.1816,-0.0312,-0.2189,-0.3201];
figure(1);
plot(X,T,'+');
title('Training Vector');
xlabel ('Input Vector P');
ylabel ('Target Vector T');
hold on;

x=-3:0.1:3;
%径向基函数1
a1=radbas(x);
figure(2);
plot(x,a1);
title('Radial Basis Transfer Function');
xlabel ('Input x');ylabel ('Output a1');
%径向基函数2
a2=radbas(x-1.5);
figure(3);
plot(x,a2);
%径向基函数3
a3=radbas(x+2);
figure(4);
plot(x,a3);
%径向基函数加权求和
a4=a1+a2*1+a3*0.5;
figure(5);
plot(x,a4);
figure(6);
plot(x,a1,'b-',x,a2,'b--',x,a3,'b--',x,a4,'m-');
legend('a1','a2','a3','a4');

%%%%%%%%%%%%RBF网络的另一种实现方式

eg=0.02;
sc1=0.01;net=newrb(X,T,eg,sc1);
Y1=net(X);
sc2=100;net=newrb(X,T,eg,sc2);
Y2=net(X);
sc3=1;net=newrb(X,T,eg,sc3);
Y3=net(X);
plot(X,Y1,'r-',X,Y2,'k-',X,Y3,'b-');legend('sample','sc=0.01','sc=100','sc=1');

%%%%%%%%%%%%%%%%%采用RBF网络进行数据聚类
clear all
load iris_dataset
[irisInputs,irisTargets]=iris_dataset;
property_1=irisInputs(1,:);
property_2=irisInputs(2,:);
property_3=irisInputs(3,:);
property_4=irisInputs(4,:);

%%%%%%%%%%%采用RBF网络进行数据拟合
%%清空环境变量
clc
clear
%%产生输入输出数据
%设置步长
interval=0.01;
%产生x1,x2
x1=-1.5:interval:1.5;
x2=-1.5:interval:1.5;
%按照函数先求的响应的函数值，作为网络的输出
F=20+x1.^2-10*cos(2*pi*x1)+x2.^2-10*cos(2*pi*x2);
%%网络建立和训练
%网络建立，输入为[x1;x2]，输出为F。spread使用默认
net=newrbe([x1;x2],F);
%%网络的效果验证
%将原数据回带，测试网络效果
ty=sim(net,[x1;x2]);
%%使用图像来看网络对非线性函数的拟合效果
figure
plot3(x1,x2,F,'rd');
hold on;
 plot3(x1,x2,ty,'b-.');
view(113,36);
title('可视化的方法观察严格的RBF神经网络的拟合效果');
xlabel('x1')
ylabel('x2')
zlabel('F')
grid on














