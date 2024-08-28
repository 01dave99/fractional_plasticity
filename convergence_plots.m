clear();

res5=readmatrix("results/frac_res_116642_0.5.csv");
res7=readmatrix("results/frac_res_116642_0.7.csv");
res9=readmatrix("results/frac_res_116642_0.9.csv");
res99=readmatrix("results/frac_res_116642_0.99.csv");
resI1=readmatrix("results/frac_res_116642_I1.csv");
resI2=readmatrix("results/frac_res_116642_I2.csv");
resI3=readmatrix("results/frac_res_116642_I3.csv");


figure(1)
hold on
xlabel("$t_n$",Interpreter="latex")
ylabel("$k$",Interpreter="latex")
plot(extract_k(res5(:,2:end)),LineWidth=1.5)
plot(extract_k(res7(:,2:end)),LineWidth=1.5)
plot(extract_k(res9(:,2:end)),LineWidth=1.5)
plot(extract_k(res99(:,2:end)),LineWidth=1.5)
legend(["$\alpha=0.5$","$\alpha=0.7$","$\alpha=0.9$","$\alpha=0.99$"],Interpreter="latex")

figure(2)
hold on
xlabel("$t_n$",Interpreter="latex")
ylabel("$k$",Interpreter="latex")
plot(extract_k(res5(:,2:end)),LineWidth=1.5)
plot(extract_k(resI1(:,2:end)),LineWidth=1.5)
plot(extract_k(resI2(:,2:end)),LineWidth=1.5)
plot(extract_k(resI3(:,2:end)),LineWidth=1.5)
legend(["$\Delta =\left( \matrix{100 & 100 \cr 100 & 200} \right)$","$\Delta =\left( \matrix{1 & 100 \cr 100 & 1000} \right)$", "$\Delta =\left( \matrix{5000 & 5000 \cr 5000 & 5000} \right)$", "$\Delta =\left( \matrix{200 & 100 \cr 100 & 100} \right)$"],Interpreter="latex")



function k=extract_k(m)
shape=size(m);
k=zeros(shape(2),1);
for i=1:shape(2)
    k(i)=length(m(m(:,i)~=0,i));
end
end