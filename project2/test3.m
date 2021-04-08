close all;
clear all;
clc;
%% ----------init-----------------------------
f = imread('./expo_building_gray.jpg');
f = mat2gray(f,[0 255]);

f_original = f;

[M,N] = size(f);

P = 2*M;
Q = 2*N;
fc = zeros(M,N);

for x = 1:1:M
    for y = 1:1:N
        fc(x,y) = f(x,y) * (-1)^(x+y);
    end
end

F_I = fft2(fc,P,Q);

figure();
subplot(1,2,1);
imshow(f,[0 1]);
xlabel('a).Original Image');

subplot(1,2,2);
imshow(log(1 + abs(F_I)),[ ]);
xlabel('b).Fourier spectrum of a).');
%% ------motion blur------------------
H = zeros(P,Q);
a = 0.02;
b = 0.02;
T = 1;
for x = (-P/2):1:(P/2)-1
     for y = (-Q/2):1:(Q/2)-1
        R = (x*a + y*b)*pi;
        if(R == 0)
            H(x+(P/2)+1,y+(Q/2)+1) = T;
        else H(x+(P/2)+1,y+(Q/2)+1) = (T/R)*(sin(R))*exp(-1i*R);
        end
     end
end

%% ------the atmospheric turbulence modle------------------
H_1 = zeros(P,Q);
k = 0.0025;
for x = (-P/2):1:(P/2)-1
     for y = (-Q/2):1:(Q/2)-1
        D = (x^2 + y^2)^(5/6);
        D_0 = 60;
        H_1(x+(P/2)+1,y+(Q/2)+1) = exp(-k*D);   
     end
end
%% -----------noise------------------
a = 0;
b = 0.2;
n_gaussian = a + b .* randn(M,N);

Noise = fft2(n_gaussian,P,Q);

figure();
subplot(1,2,1);
imshow(n_gaussian,[-1 1]);
xlabel('a).Gaussian noise');

subplot(1,2,2);
imshow(log(1 + abs(Noise)),[ ]);
xlabel('b).Fourier spectrum of a).');
%%
G = H .* F_I + Noise;
% G = H_1 .* F_I + Noise;
gc = ifft2(G);

gc = gc(1:1:M+27,1:1:N+27);
for x = 1:1:(M+27)
    for y = 1:1:(N+27)
        g(x,y) = gc(x,y) .* (-1)^(x+y);
    end
end

gc = gc(1:1:M,1:1:N);
for x = 1:1:(M)
    for y = 1:1:(N)
        g(x,y) = gc(x,y) .* (-1)^(x+y);
    end
end

figure();
subplot(1,2,1);
imshow(f,[0 1]);
xlabel('a).Original Image');

subplot(1,2,2);
imshow(log(1 + abs(F_I)),[ ]);
xlabel('b).Fourier spectrum of a).');

figure();
subplot(1,2,1);
imshow(abs(H),[ ]);
xlabel('c).The motion modle H(u,v)(a=0.02,b=0.02,T=1)');

subplot(1,2,2);
n = 1:1:P;
plot(n,abs(H(400,:)));
axis([0 P 0 1]);grid; 
xlabel('H(n,400)');
ylabel('|H(u,v)|');

figure();
subplot(1,2,1);
imshow(real(g),[0 1]);
xlabel('d).Result image');

subplot(1,2,2);
imshow(log(1 + abs(G)),[ ]);
xlabel('e).Fourier spectrum of d). ');
%% --------------inverse_filtering---------------------
%F = G ./ H;
%F = G ./ H_1;

for x = (-P/2):1:(P/2)-1
     for y = (-Q/2):1:(Q/2)-1
        D = (x^2 + y^2)^(0.5);
        if(D < 258) 
            F(x+(P/2)+1,y+(Q/2)+1) = G(x+(P/2)+1,y+(Q/2)+1) ./ H_1(x+(P/2)+1,y+(Q/2)+1);
        % no noise D < 188
        % noise D < 56
        else F(x+(P/2)+1,y+(Q/2)+1) = G(x+(P/2)+1,y+(Q/2)+1);
        end 
     end
end

% Butterworth_Lowpass_Filters
H_B = zeros(P,Q);
D_0 = 70;
for x = (-P/2):1:(P/2)-1
     for y = (-Q/2):1:(Q/2)-1
        D = (x^2 + y^2)^(0.5);
        %if(D < 200) H_B(x+(P/2)+1,y+(Q/2)+1) = 1/(1+(D/D_0)^100);end 
        H_B(x+(P/2)+1,y+(Q/2)+1) = 1/(1+(D/D_0)^20);
     end
end

F = F .* H_B;

f = real(ifft2(F));
f = f(1:1:M,1:1:N);

for x = 1:1:(M)
    for y = 1:1:(N)
        f(x,y) = f(x,y) * (-1)^(x+y);
    end
end
%% ------show Result------------------
figure();
subplot(1,2,1);
imshow(f,[0 1]);
xlabel('a).Result image');

subplot(1,2,2);
imshow(log(1 + abs(F)),[ ]);
xlabel('b).Fourier spectrum of a).');

figure();
n = 1:1:P;
plot(n,abs(F(400,:)),'r-',n,abs(F(400,:)),'b-');
axis([0 P 0 1000]);grid; 
xlabel('Number of rows(400th column)');
ylabel('Fourier amplitude spectrum');
legend('F_{limit}(u,v)','F(u,v)');

figure();
n = 1:1:P;
plot(n,abs(H(400,:)),'g-');
axis([0 P 0 1]);grid; 
xlabel('H''_{s}(n,400)');
ylabel('|H''_{s}(u,v)|');
%% ----------Wiener filters-----------
% K = 0.000014;
K = 0.02;
%H_Wiener = ((abs(H_1).^2)./((abs(H_1).^2)+K)).*(1./H_1);
H_Wiener = ((abs(H).^2)./((abs(H).^2)+K)).*(1./H);

F_Wiener = H_Wiener .*  G;
f_Wiener = real(ifft2(F_Wiener));
f_Wiener = f_Wiener(1:1:M,1:1:N);

for x = 1:1:(M)
    for y = 1:1:(N)
        f_Wiener(x,y) = f_Wiener(x,y) * (-1)^(x+y);
    end
end

[SSIM_Wiener mssim] = ssim_index(f_Wiener,f_original,[0.01 0.03],ones(8),1);
SSIM_Wiener 
%% ------show Result------------------
figure();
subplot(1,2,1);
%imshow(f_Wiener(1:128,1:128),[0 1]);
imshow(f_Wiener,[0 1]);
xlabel('d).Result image by Wiener filter');

subplot(1,2,2);
imshow(log(1+abs(F_Wiener)),[ ]);
xlabel('c).Fourier spectrum of c).');
% subplot(1,2,2);
% %imshow(f(1:128,1:128),[0 1]);
% imshow(f,[0 1]);
% xlabel('e).Result image by inverse filter');


figure();
n = 1:1:P;
plot(n,abs(F(400,:)),'r-',n,abs(F_Wiener(400,:)),'b-');
axis([0 P 0 500]);grid; 
xlabel('Number of rows(400th column)');
ylabel('Fourier amplitude spectrum');
legend('F(u,v)','F_{Wiener}(u,v)');

figure();
subplot(1,2,1);
imshow(log(1 + abs(H_Wiener)),[ ]);
xlabel('a).F_{Wiener}(u,v).');

subplot(1,2,2);
n = 1:1:P;
plot(n,abs(H_Wiener(400,:)));
axis([0 P 0 80]);grid; 
xlabel('Number of rows(400th column)');
ylabel('Amplitude spectrum');

%% ------------Constrained_least_squares_filtering---------
p_laplacian = zeros(M,N);
Laplacian = [ 0 -1  0;
             -1  4 -1;
              0 -1  0];
p_laplacian(1:3,1:3) = Laplacian;        

P = 2*M;
Q = 2*N;
for x = 1:1:M
    for y = 1:1:N
        p_laplacian(x,y) = p_laplacian(x,y) * (-1)^(x+y);
    end
end
P_laplacian = fft2(p_laplacian,P,Q);

F_C = zeros(P,Q);
r = 0.2;
H_clsf = ((H')./((abs(H).^2)+r.*P_laplacian));

F_C = H_clsf .* G;

f_c = real(ifft2(F_C));
f_c = f_c(1:1:M,1:1:N);

for x = 1:1:(M)
   for y = 1:1:(N)
       f_c(x,y) = f_c(x,y) * (-1)^(x+y);
    end
end

%%  
figure();
subplot(1,2,1);
imshow(f_c,[0 1]);
xlabel('e).Result image by constrained least squares filter (r = 0.2)');

subplot(1,2,2);
imshow(log(1 + abs(F_C)),[ ]);
xlabel('f).Fourier spectrum of c).');

[SSIM_CLSF mssim] = ssim_index(f_c,f_original,[0.01 0.03],ones(8),1);

figure();
subplot(1,2,1);
imshow(log(1 + abs(H_clsf)),[ ]);
xlabel('a).F_{clsf}(u,v).');

subplot(1,2,2);
n = 1:1:P;
plot(n,abs(H_clsf(400,:)));
axis([0 P 0 80]);grid; 
xlabel('Number of rows(400th column)');
ylabel('Amplitude spectrum');

%%
m = [-1 0 1;-2 0 2;-1 0 1];
H = freqz2(m,602,602);

%%


%=============================================================================

inimg = imread('cameraman.tif');

subplot(131)

imshow(inimg), title('Original image')

[M,N] = size(inimg);               % Original image size

%====================================================================

h = fspecial('gaussian',25,4);  % Gaussian filter

%====================================================================

% 空域滤波

gx = imfilter(inimg,h,'same','replicate');  % 空域图像滤波

subplot(132)

imshow(gx,[]);title('Spatial domain filtering')

%====================================================================

%    频域滤波

%====================================================================

h_hf = floor(size(h)/2);                  % 空域滤波器半高/宽

imgp = padarray(inimg, [h_hf(1),h_hf(2)],'replicate'); % Padding boundary with copying pixels

% PQ = paddedsize(size(imgp));   % Gonzalez DIP教材提供的函数，非MATLAB内部函数

PQ  = 2*size(imgp);

Fp = fft2(double(imgp), PQ(1), PQ(2));     % 延拓图像FFT

% h  = rot90(h,2);  % Mask旋转180度，非对称h需此步骤！因频域乘积对应空域卷积，而空域滤波为相关。

P = PQ(1); Q = PQ(2);

center_h = h_hf+1;                     % 空域小模板h中心位置

hp = zeros(P,Q);                          % 预分配内存，产生P×Q零矩阵

hp(1:size(h,1),1:size(h,2)) = h;  % h置于hp左上角

hp = circshift(hp,[-(center_h(1)-1),-(center_h(2)-1)]); % 循环移位，h中心置于hp左上角

%====================================================================

Hp = fft2(double(hp));                 % hp滤波器做FFT

%====================================================================

Gp = Hp.*Fp;                                %  频域滤波

gp = real(ifft2(Gp));                     % 反变换，取实部

gf = gp(h_hf(1)+1:M+ h_hf(1),  h_hf(2)+1:N + h_hf(2));  % 截取有效数据

subplot(133)

imshow(uint8(gf),[]),  title('Frequency domain filtering')

% 注：以上处理中，频域图像Fp与滤波器Hp均未中心化，因此，返回空域时无需反中心化。

% 另外，直接调用Hp = freqz2(h,P,Q)获得的2D频域响应，则是中心化的。

%%
N=256;
n=.2;
f=imread('../images/lena.jpg',N,N);
figure(1)
imagesc(f)
colormap(gray)
b=ones(4,4)/4^2;
F=fft2(f);
B=fft2(b,N,N);
G=F.*B;
g=ifft2(G)+10*randn(N,N);
G=fft2(g);
figure(2)
imagesc(abs(ifft2(G)))
colormap(gray)
BF=find(abs(B)<n);
%B(BF)=max(max(B))/1.5;
B(BF)=n;
H=ones(N,N)./B;
I=G.*H;
im=abs(ifft2(I));
figure(3)
imagesc(im)
colormap(gray)
%%
clc;
clear all;
close all;
f=im2double(imread('cameraman.tif'));
f=imresize(f,[256 256])
figure,(imshow(f))
[M,N]=size(f);
% k=2.5;
%  for i=1:size(f,1)
%      for j=1:size(f,2)
%          h(i,j)=exp((-k)*((i-M/2)^2+(j-N/2)^2)^(5/6));
%       end
%  end
h=fspecial('gaussian',260,2);
g=(imfilter(f,h,'circular'));
figure,imshow(g,[]);

G = fftshift(fft2(g));
figure,imshow(log(abs(G)),[]);

H = fftshift(fft2(h));
figure,imshow(log(abs(H)),[]);

F = zeros(size(f));
R=70;
for u=1:size(f,2)
    for v=1:size(f,1)
        du = u - size(f,2)/2;
        dv = v - size(f,1)/2;
        if du^2 + dv^2 <= R^2;
        F(v,u) = G(v,u)./H(v,u);
        end
    end
end

figure,imshow(log(abs(F)),[]);

% fRestored = abs(ifft2(ifftshift(F)));
fRestored = abs(ifftshift(ifft2(F)));
figure,imshow(fRestored, []);

%%
%%
close all; clear;
f= imread('expo_building_gray.jpg');

figure;
imshow(f);
title("original");
[m,n]=size(f);

h=fspecial('gaussian',5,2); %gaussian spatial filter
g=imfilter(f,h,'circular'); %filter on spatial domain
noise = uint8(randn(m,n));
g = g+noise;
figure,imshow(g,[]); 
title("with blur and noise");

G = fftshift(fft2(g)); 
figure;
P = abs(G).^2;
imagesc(log(1+P));
title("power spectrum");

[x,y] = meshgrid(1:n, 1:m);
figure;
mesh(x,y,log(1+P));
title("3d power");
view(10,20);
axis off;

H = fftshift(fft2(h,size(G,1), size(G,2)));

%%
F = zeros(size(f)); %pixels far away are 0 by default
R=200;
for v=1:size(f,1)
    for u=1:size(f,2)
        du = u - size(f,2)/2;
        dv = v - size(f,1)/2;
        if du^2 + dv^2 <= R^2
            F(v,u) = G(v,u)./H(v,u);
        end
    end
end

figure;
imshow(log(abs(F)),[]);
f_h = abs(ifft2(F));
figure;
imshow(f_h,[]);
title('restored image');