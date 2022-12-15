clc;
clear;
L = 512;
m = 128;
N_z = 128;
x = zeros(L,1);

% ------------------------ a) ------------------------

A = randperm(L);
A=A(1:N_z);
x(A) = sign(randn(1, N_z));
figure; 
subplot(3,1,1);
plot(x);
axis([0 L -2 2]);
title('Data');

% ------------------------ b) ------------------------

A = randn(N_z, L);
b = A * x + 0.005 * randn(m,1);
subplot(3,1,2);
plot(1:m, b, 'g');
title('Compressed data');

% ------------------------ c) ------------------------
Uncompressed_data = l1eq_pd(x, A,[], b); 
subplot(3,1,3);
plot(1:L, Uncompressed_data, 'r');
axis([0 512 -2 2]);
title('Recovered Data');


function l1 = l1eq_pd(x0, A, A1, b)
t_cg = 0.00000001;
m_cg = 200;
t_pd = 0.001;
m_pd = 50;
if (nargin < 5), t_pd = t_pd;  end
if (nargin < 6), m_pd = m_pd;  end
if (nargin < 7), t_cg = t_cg;  end
if (nargin < 8), m_cg = m_cg;  end

N = length(x0);
y_0 = zeros(N,1);
y_1 = ones(N,1);
beta = 0.5;
U = 10;
g = [y_0; y_1];

% Checking initial point
if (isa(A,'function_handle')) & (norm(A*x0-b)/norm(b) > t_cg)
    disp('hard to find initial point');
elseif (isa(A,'function_handle')) & (r_cg > 1/2)
    disp('no initial point');
    l1 = x0;
    return;
elseif (norm(A*x0-b)/norm(b) > t_cg)
    disp('Hard to find intial point');
    opts.POSDEF = true; opts.SYM = true;
    [w, con] = linsolve(A*A', b, opts);
    if (con < 1e-14)
      disp('cant find initial point');
      l1 = x0;
      return;
    end  
end
x = x0;
u = (0.95)*abs(x0) + (0.10)*max(abs(x0));

%first iteration
f1 = x - u;
f2 = -x - u;
l_1 = -1./f1;
l_2 = -1./f2;
if (isa(A,'function_handle'))
  v = -A(l_1-l_2);
  A1v = A1(v);
  perim = A(x) - b;
else
  v = -A*(l_1-l_2);
  A1v = A'*v;
  perim = A*x - b;
end

gap = -(f1'*l_1 + f2'*l_2);
t = U*2*N/gap;
dual = g + [l_1-l_2; -l_1-l_2] + [A1v; y_0];
percen = [-l_1.*f1; -l_2.*f2] - (1/t);
r_norm = norm([dual; percen; perim]);
pr = 0;
Output = (gap < t_pd) | (pr >= m_pd);
while (~Output)
  
  pr = pr + 1;

  s1 = -l_1./f1 - l_2./f2;
  s2 = l_1./f1 - l_2./f2;
  sx = s1 - s2.^2./s1;
  e1 = -1/t*(-1./f1 + 1./f2) - A1v;
  e2 = -1 - 1/t*(1./f1 + 1./f2);
  e3 = -perim;
  
  if (isa(A,'function_handle')) & (r_cg > 1/2)
    e_0 = e3 - A(e1./sx - e2.*s2./(sx.*s1));
    hpfun = @(z) -A(1./sx.*A1(z));
    [dv, r_cg, i_cg] = cgsolve(hpfun, e_0, t_cg, m_cg, 0);
    
    disp('Cant find solution.');
    l1 = x;
    dx = (e1 - e2(s2)./s1 - A1(dv))./sx;
    Adx = A*dx;
    A1dv = A1*dv;
    return
  else
    e_0 = -(e3 - A*(e1./sx - e2.*s2./(sx.*s1)));
    hp = A*(sparse(diag(1./sx))*A');
    [dv,con] = linsolve(hp, e_0); % ------- Quan Wang
    if (con < 1e-14)
      disp('Previous iteration.)');
      l1 = x;
      return
    end
    dx = (e1 - e2.*s2./s1 - A'*dv)./sx;
    Adx = A*dx;
    A1dv = A'*dv;
  end
  
  du = (e2 - s2.*dx)./s1;
  
  d1 = (l_1./f1).*(-dx+du) - l_1 - (1/t)*1./f1;
  d2 = (l_2./f2).*(dx+du) - l_2 - 1/t*1./f2;
  
  % m
  i1 = find(d1 < 0);  i2 = find(d2 < 0);
  s = min([1; -l_1(i1)./d1(i1); -l_2(i2)./d2(i2)]);
  i1 = find((dx-du) > 0);  i2 = find((-dx-du) > 0);
  s = (0.99)*min([s; -f1(i1)./(dx(i1)-du(i1)); -f2(i2)./(-dx(i2)-du(i2))]);
  
  back1 = 0;
  while (1)
    l1 = x + s*dx;  up = u + s*du; 
    l_1p = l_1 + s*d1;  l_2p = l_2 + s*d2;
    f1p = l1 - up;  f2p = -l1 - up;  
  
    r3 = perim + s*Adx;
    
    s = beta*s;
    back1 = back1 + 1;
    if (back1 > 32)
      disp('Stuck..')
      l1 = x;
      return
    end
  end


  
  %next iteration
  x = l1;  u = up;
  A1v = A1l2; 
  l_1 = l_1p;  l_2 = l_2p;
  f1 = f1p;  f2 = f2p;
  
  gap = -(f1'*l_1 + f2'*l_2);
  t = U*2*N/gap;
  perim = r3;
  percen = [-l_1.*f1; -l_2.*f2] - (1/t);
  dual = g + [l_1-l_2; -l_1-l_2] + [A1v; y_0];
  r_norm = norm([dual; percen; perim]);
  
  Output = (gap < t_pd) | (pr >= m_pd);
end

end 