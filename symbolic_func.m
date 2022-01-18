% syms x1 x2 real
% x = [x1;x2]; % column vector of symbolic variables
% f = log(1 + 3*(x2 - (x1^3 - x1))^2 + (x1 - 4/3)^2)
% fsurf(f,[-2 2],'ShowContours','on')
% view(127,38)
% gradf = jacobian(f,x).' % column gradf
% hessf = jacobian(gradf,x)
% fh = matlabFunction(f,gradf,hessf,'vars',{x});
% options = optimoptions('fminunc', ...
%     'SpecifyObjectiveGradient', true, ...
%     'HessianFcn', 'objective', ...
%     'Algorithm','trust-region', ...
%     'Display','final');
% [xfinal,fval,exitflag,output] = fminunc(fh,[-1;2],options)
syms x1 x2 x3 x4 x5 x6 x7 x8 real
x = [x1;x2;x3;x4;x5;x6;x7;x8];
k=10;
f = (1/k)/(1/k+x1) + (1/k + x1/2)/(1/k + x1/2 + x2) + (1/k + x2/2)/(1/k + x2/2 + x3) + (1/k + x3/2)/(1/k + x3/2 + x4) ...
    + (1/k + x4/2)/(1/k + x4/2 + x5) + (1/k + x5/2)/(1/k + x5/2 + x6) + (1/k + x6/2)/(1/k + x6/2 + x7) + (1/k + x7/2)/(1/k + x7/2 + x8) + (1/k + x8/2)/(1/k + x8/2 + 1);
gradf = jacobian(f,x);
hessf = jacobian(gradf,x);
fh = matlabFunction(f,gradf,hessf,'vars',{x});
c1 = x1-1;
c2 = -x1;
c3 = x2-1;
c4 = -x2;
c5 = x3-1;
c6 = -x3;
c7 = x4-1;
c8 = -x4;
c9 = x5-1;
c10 = -x5;
c11 = x6-1;
c12 = -x6;
c13 = x7-1;
c14 = -x7;
c15 = x8-1;
c16 = -x8;
c = [c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16];
gradc = jacobian(c,x).'; % transpose to put in correct form
constraint = matlabFunction(c,[],gradc,[],'vars',{x});

hessc1 = jacobian(gradc(:,1),x); % constraint = first c column
hessc2 = jacobian(gradc(:,2),x);
hessc3 = jacobian(gradc(:,3),x); % constraint = first c column
hessc4 = jacobian(gradc(:,4),x);
hessc5 = jacobian(gradc(:,5),x); % constraint = first c column
hessc6 = jacobian(gradc(:,6),x);
hessc7 = jacobian(gradc(:,7),x); % constraint = first c column
hessc8 = jacobian(gradc(:,8),x);
hessc9 = jacobian(gradc(:,9),x); % constraint = first c column
hessc10 = jacobian(gradc(:,10),x);
hessc11 = jacobian(gradc(:,11),x); % constraint = first c column
hessc12 = jacobian(gradc(:,12),x);
hessc13 = jacobian(gradc(:,13),x); % constraint = first c column
hessc14 = jacobian(gradc(:,14),x);
hessc15 = jacobian(gradc(:,15),x); % constraint = first c column
hessc16 = jacobian(gradc(:,16),x);

hessfh = matlabFunction(hessf,'vars',{x});
hessc1h = matlabFunction(hessc1,'vars',{x});
hessc2h = matlabFunction(hessc2,'vars',{x});
hessc3h = matlabFunction(hessc3,'vars',{x});
hessc4h = matlabFunction(hessc4,'vars',{x});
hessc5h = matlabFunction(hessc5,'vars',{x});
hessc6h = matlabFunction(hessc6,'vars',{x});
hessc7h = matlabFunction(hessc7,'vars',{x});
hessc8h = matlabFunction(hessc8,'vars',{x});
hessc9h = matlabFunction(hessc9,'vars',{x});
hessc10h = matlabFunction(hessc10,'vars',{x});
hessc11h = matlabFunction(hessc11,'vars',{x});
hessc12h = matlabFunction(hessc12,'vars',{x});
hessc13h = matlabFunction(hessc13,'vars',{x});
hessc14h = matlabFunction(hessc14,'vars',{x});
hessc15h = matlabFunction(hessc15,'vars',{x});
hessc16h = matlabFunction(hessc16,'vars',{x});

myhess = @(x,lambda)(hessfh(x) + ...
    lambda.ineqnonlin(1)*hessc1h(x) + ...
    lambda.ineqnonlin(2)*hessc2h(x) + ...
    lambda.ineqnonlin(3)*hessc3h(x) + ...
    lambda.ineqnonlin(4)*hessc4h(x) + ...
    lambda.ineqnonlin(5)*hessc5h(x) + ...
    lambda.ineqnonlin(6)*hessc6h(x) + ...
    lambda.ineqnonlin(7)*hessc7h(x) + ...
    lambda.ineqnonlin(8)*hessc8h(x) + ...
    lambda.ineqnonlin(9)*hessc9h(x) + ...
    lambda.ineqnonlin(10)*hessc10h(x) + ...
    lambda.ineqnonlin(11)*hessc11h(x) + ...
    lambda.ineqnonlin(12)*hessc12h(x) + ...
    lambda.ineqnonlin(13)*hessc13h(x) + ...
    lambda.ineqnonlin(14)*hessc14h(x) + ...
    lambda.ineqnonlin(15)*hessc15h(x) + ...
    lambda.ineqnonlin(16)*hessc16h(x));

options = optimoptions('fmincon', ...
    'Algorithm','interior-point', ...
    'SpecifyObjectiveGradient',true, ...
    'SpecifyConstraintGradient',true, ...
    'HessianFcn',myhess, ...
    'StepTolerance', 1.0000e-30, ...
    'ConstraintTolerance', 1.0000e-30, ...
    'OptimalityTolerance', 1.0000e-30, ...
    'ObjectiveLimit', -1.0000e+20, ...
    'Display','final');
% fh2 = objective without Hessian
fh2 = matlabFunction(f,gradf,'vars',{x});
[xfinal,fval,exitflag,output] = fmincon(fh2,[1/k;1/k;1/k;1/k;1/k;1/k;1/k;1/k],...
    [],[],[],[],[],[],constraint,options)