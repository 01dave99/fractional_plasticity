clear();
m5=readmatrix("defl_mid_0.5.csv");
m7=readmatrix("defl_mid_0.7.csv");
m9=readmatrix("defl_mid_0.9.csv");
m99=readmatrix("defl_mid_0.99.csv");

r5=readmatrix("defl_right_0.5.csv");
r7=readmatrix("defl_right_0.7.csv");
r9=readmatrix("defl_right_0.9.csv");
r99=readmatrix("defl_right_0.99.csv");

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
