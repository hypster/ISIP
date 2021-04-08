close all;clear;
f = imread('./expo_building_gray.jpg');
f = double(f);
[m,n] = size(f);
F = fft2(f);
% S = fftshift(log(1+abs(F)));
% imshow(S,[]);
%%
h = fspecial('sobel')';
freqz2(h);

[m,n] = size(f);
H = freqz2(h, 2*n, 2*m);
H1 = ifftshift(H);
imshow(abs(H),[]);
figure;
imshow(abs(H1), []);
gs = imfilter(f, h);
imshow(gs,[]);
%%
F = fft2(f, size(H1,1), size(H1,2));
gf = ifft2(F.*H1);
gf = gf(1:size(f,1), 1:size(f,2));
figure;
imshow(gf,[]);

%%
m = 8;
n = 5;
u = -m/2:m/2-1;
v = floor(-n/2)+1:n/2;

[x,y] = meshgrid(v,u);
z = x.^2 + y.^2

%%
[x,y] = meshgrid(1:n, 1:m);
nf = @(x,y) 100*sin(2*pi*50*x/m + 2*pi*100*y/n);
noise = nf(x,y);
fn = f+noise;
figure;
imshow(f,[]);
figure;
imshow(fn,[]);
%%
F = fftshift(fft2(f));
figure;
imagesc(log(1+abs(F)));
FN = fftshift(fft2(fn));
figure;
imagesc(log(1+abs(FN)));
%%
C = [401 702;289 527];
C = [290-35 526+35;398 701];
C(:,1) = C(:,1) - m/2-1;
C(:,2) = C(:,2) - n/2-1;
H = notchfilter(C(1,:), m,n);
HR = 1 - H;
figure;
imagesc(log(1+abs(FN.*HR)))
%%

figure;
fdn = abs(ifft2(HR.*FN));
imshow(fdn,[]);
%%

function h = notchfilter(f0, n1,n2)

% close all; clear all; clc
% n1 = 1080; n2 = 1080; f0 = 80;
% [k1 k2] = meshgrid(-round(n2/2)+1:round(n2/2), -round(n1/2)+1:round(n1/2));
[x,y] = meshgrid(n2, n1);
k1 = fftshift(x);
k2 = fftshift(y);
cx = f0(1);
cy = f0(2);
d1 = sqrt((k1-cx).^2 + (k2-cy).^2);
d2 = sqrt((k1+cx).^2 + (k2+cy).^2);
d0 = 30;
h = zeros(n1,n2);
h(d1 < d0) = 1;
h(d2 < d0) = 1;

figure;imshow(h)

end