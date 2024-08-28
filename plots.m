clear();
m5=readmatrix("results/defl_mid_116642_0.5.csv");
m7=readmatrix("results/defl_mid_116642_0.7.csv");
m9=readmatrix("results/defl_mid_116642_0.9.csv");
m99=readmatrix("results/defl_mid_116642_0.99.csv");

r5=readmatrix("results/defl_right_116642_0.5.csv");
r7=readmatrix("results/defl_right_116642_0.7.csv");
r9=readmatrix("results/defl_right_116642_0.9.csv");
r99=readmatrix("results/defl_right_116642_0.99.csv");

mI1=readmatrix("results/defl_mid_116642_I1.csv");
mI2=readmatrix("results/defl_mid_116642_I2.csv");
mI3=readmatrix("results/defl_mid_116642_I3.csv");

rI1=readmatrix("results/defl_right_116642_I1.csv");
rI2=readmatrix("results/defl_right_116642_I2.csv");
rI3=readmatrix("results/defl_right_116642_I3.csv");

figure(1)
hold on
plot(m5,LineWidth=1)
plot(m7,LineWidth=1)
plot(m9,LineWidth=1)
plot(m99,LineWidth=1)
xlabel("$t$",Interpreter="latex")
ylabel("$d_y$",Interpreter="latex")
legend(["$\alpha=0.5$","$\alpha=0.7$","$\alpha=0.9$","$\alpha=0.99$"],Interpreter="latex")

figure(2)
hold on
plot(r5,LineWidth=1)
plot(r7,LineWidth=1)
plot(r9,LineWidth=1)
plot(r99,LineWidth=1)
xlabel("$t$",Interpreter="latex")
ylabel("$d_x$",Interpreter="latex")
legend(["$\alpha=0.5$","$\alpha=0.7$","$\alpha=0.9$","$\alpha=0.99$"],Interpreter="latex")

figure(3)
hold on
plot(r5-r99,LineWidth=1)
plot(r7-r99,LineWidth=1)
plot(r9-r99,LineWidth=1)
plot(r99-r99,LineWidth=1)
xlabel("$t$",Interpreter="latex")
ylabel("$d_x-d_x^{99}$",Interpreter="latex")
legend(["$\alpha=0.5$","$\alpha=0.7$","$\alpha=0.9$","$\alpha=0.99$"],Interpreter="latex")

figure(4)
hold on
plot(m5-m99,LineWidth=1)
plot(m7-m99,LineWidth=1)
plot(m9-m99,LineWidth=1)
plot(m99-m99,LineWidth=1)
xlabel("$t$",Interpreter="latex")
ylabel("$d_y-d_y^{99}$",Interpreter="latex")
legend(["$\alpha=0.5$","$\alpha=0.7$","$\alpha=0.9$","$\alpha=0.99$"],Interpreter="latex")

figure(5)
hold on
plot(m5,LineWidth=1)
plot(mI1,LineWidth=1)
plot(mI2,LineWidth=1)
plot(mI3,LineWidth=1)
xlabel("$t$",Interpreter="latex")
ylabel("$d_y$",Interpreter="latex")
legend(["$\Delta =\left( \matrix{100 & 100 \cr 100 & 200} \right)$","$\Delta =\left( \matrix{1 & 100 \cr 100 & 1000} \right)$", "$\Delta =\left( \matrix{5000 & 5000 \cr 5000 & 5000} \right)$", "$\Delta =\left( \matrix{200 & 100 \cr 100 & 100} \right)$"],Interpreter="latex")

figure(6)
hold on
plot(r5,LineWidth=1)
plot(rI1,LineWidth=1)
plot(rI2,LineWidth=1)
plot(rI3,LineWidth=1)
xlabel("$t$",Interpreter="latex")
ylabel("$d_x$",Interpreter="latex")
legend(["$\Delta =\left( \matrix{100 & 100 \cr 100 & 200} \right)$","$\Delta =\left( \matrix{1 & 100 \cr 100 & 1000} \right)$", "$\Delta =\left( \matrix{5000 & 5000 \cr 5000 & 5000} \right)$", "$\Delta =\left( \matrix{200 & 100 \cr 100 & 100} \right)$"],Interpreter="latex")

figure(7)
hold on
plot(m5-m99,LineWidth=1)
plot(mI1-m99,LineWidth=1)
plot(mI2-m99,LineWidth=1)
plot(mI3-m99,LineWidth=1)
xlabel("$t$",Interpreter="latex")
ylabel("$d_y-d_y^{99}$",Interpreter="latex")
legend(["$\Delta =\left( \matrix{100 & 100 \cr 100 & 200} \right)$","$\Delta =\left( \matrix{1 & 100 \cr 100 & 1000} \right)$", "$\Delta =\left( \matrix{5000 & 5000 \cr 5000 & 5000} \right)$", "$\Delta =\left( \matrix{200 & 100 \cr 100 & 100} \right)$"],Interpreter="latex")

figure(8)
hold on
plot(r5-r99,LineWidth=1)
plot(rI1-r99,LineWidth=1)
plot(rI2-r99,LineWidth=1)
plot(rI3-r99,LineWidth=1)
xlabel("$t$",Interpreter="latex")
ylabel("$d_x-d_x^{99}$",Interpreter="latex")
legend(["$\Delta =\left( \matrix{100 & 100 \cr 100 & 200} \right)$","$\Delta =\left( \matrix{1 & 100 \cr 100 & 1000} \right)$", "$\Delta =\left( \matrix{5000 & 5000 \cr 5000 & 5000} \right)$", "$\Delta =\left( \matrix{200 & 100 \cr 100 & 100} \right)$"],Interpreter="latex")