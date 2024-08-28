clear();
y0=10000;
d1=100;
d2=200;
d3=100;
n=150;
d=2;
alpha=0.5;
m=20;
X1=zeros(1,m-1);
Y1=zeros(1,m-1);
U1=zeros(1,m-1);
V1=zeros(1,m-1);
rU1=zeros(1,m-1);
rV1=zeros(1,m-1);

X2=zeros(1,m-1);
Y2=zeros(1,m-1);
U2=zeros(1,m-1);
V2=zeros(1,m-1);
rU2=zeros(1,m-1);
rV2=zeros(1,m-1);

ws=gamma((0:n)+1-alpha)./(gamma(1-alpha)*(factorial(0:n)));
h=[d1/n d3/n; d3/n d2/n];


for j=1:m-1
    sig1=-sqrt(2*y0)+2*sqrt(2*y0)*j/m;
    sigma=[ sig1 0; 0 sig1+sqrt(2*y0)];

df=zeros(d,d);
for i=0:n
    df=df+(grad(sigma,i,h)*ws(i+1)).*(h.^(1-alpha));
end
df=df/norm(df,"fro");
rdf=dev(sigma)/norm(dev(sigma),"fro");
X1(j)=sigma(1,1);
Y1(j)=sigma(2,2);
U1(j)=df(1,1);
V1(j)=df(2,2);
rU1(j)=rdf(1,1);
rV1(j)=rdf(2,2);

end

for j=1:m-1
    sig1=2*sqrt(2*y0)*j/m;
    sigma=[ sig1 0; 0 sig1-sqrt(2*y0)];


df=zeros(d,d);
for i=0:n
    df=df+(grad(sigma,i,h)*ws(i+1)).*(h.^(1-alpha));
end
df=df/norm(df,"fro");
rdf=dev(sigma)/norm(dev(sigma),"fro");
X2(j)=sigma(1,1);
Y2(j)=sigma(2,2);
U2(j)=df(1,1);
V2(j)=df(2,2);
rU2(j)=rdf(1,1);
rV2(j)=rdf(2,2);
end
figure(1)
quiver([X1 X2],[Y1 Y2],[U1 U2],[V1 V2])
axis equal
grid on
xlabel("$\sigma_{11}$",Interpreter="latex")
ylabel("$\sigma_{22}$",Interpreter="latex")
legend("$\alpha=0.5$", "$\alpha=1$",Interpreter="latex")
ax=gca;
ax.XTickLabel={};
ax.YTickLabel={};
hold on
quiver([X1 X2],[Y1 Y2],[rU1 rU2],[rV1 rV2])
m=0:100;
X1=-sqrt(2*y0)+2*sqrt(2*y0)*m/length(m);
Y1=X1;
Y2=X1+sqrt(2*y0);
plot(X1+sqrt(2*y0),Y1,X1,Y2,Color=[0.9290 0.6940 0.1250])

legend("$\alpha=0.5$", "$\alpha=1$","yield surface",Interpreter="latex")





function df = grad(sigma,k,h)
d=2;
df=zeros(d,d);
for i = 1:d
    for j = 1:d
        sigij1=sigma;
        sigij1(i,j)=sigma(i,j)-k*h(i,j);
        tmp1=dev(sigij1)/norm(dev(sigij1),"fro");
        sigij2=sigma;
        sigij2(i,j)=sigma(i,j)+k*h(i,j);
        tmp2=dev(sigij2)/norm(dev(sigij2),"fro");
        df(i,j)=(tmp1(i,j)+tmp2(i,j))/2;
    end
end
end

function dsig = dev(sigma)
d=2;
    dsig=sigma-trace(sigma)/d*eye(d);
end

