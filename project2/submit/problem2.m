close all;clear;
f = imread('./expo_building_gray.jpg');
f = double(f);
[m,n] = size(f);


%% add noise
[x,y] = meshgrid(1:n, 1:m);
nf = @(x,y) 100*sin(2*pi*50*x/m + 2*pi*100*y/n);
noise = nf(x,y);
fn = f+noise;
figure;
imshow(f,[]);
title("original");
saveas(gcf, "p2_original.jpg");
figure;
imshow(fn,[]);
title("with noise");
saveas(gcf, "p2_with_noise.jpg");
%% fft transform
F = fft2(f);
S = fftshift(log(1+abs(F)));
figure;
imagesc(S);
title("spectrum");
saveas(gcf, "p2_spectrum.jpg");
FN = fftshift(fft2(fn));
figure;
imagesc(log(1+abs(FN)));
title("spectrum with noise");
saveas(gcf, "p2_spectrum_noise.jpg");

%% power spectrum
figure;
p = abs(FN).^2;
imagesc(log(1+p));
title("2d power");
saveas(gcf, "p2_power_2d.jpg");
[x,y] = meshgrid(1:n,1:m);
figure;
mesh(x,y,log(1+p));
view(10,20);
axis off;
title("3d power");
saveas(gcf, "p2_power_3d.jpg");
figure;
plot(1:m, log(1+p(:,702)'));
title("1d along y=702");
saveas(gcf,"p2_power_1d.jpg");

%% noise location found in the frequency plot
fs = [702 401;527 289]; %center of location in x,y pair
ss = [20 20]; %sigma 

filter = ones(m,n);
for i = 1:size(fs,1)
    d2 = (x-fs(i,1)).^2 + (y-fs(i,2)).^2;
    filter = filter.*(1 - exp(-d2./(2*ss(i)^2))); %this is a gaussian notch reject filter
end

G = filter.*FN;
figure;
imagesc(log(1+abs(G)));

f2 = abs(ifft2(G)); %processed graph
figure;
imshow(f2,[]);
title("after restoration");
saveas(gcf, "p2_after_restoration.jpg");
%%
figure;
imagesc(log(abs(G).^2));
title("denoised power spectrum");
saveas(gcf, "power_denoise.jpg");
