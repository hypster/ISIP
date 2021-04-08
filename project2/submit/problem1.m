%% cosine function with frequency 1
f = @(x,y) cos(2*pi*x + 2*pi*y); 

[x,y] = meshgrid(0:0.01:2,0:0.01:2);
z = f(x,y);
figure;
surf(x,y,z);
title("3d plot of cos(2*pi*x + 2*pi*y)");
saveas(gcf, "./result/cosine.jpg");
figure;
imshow(z,[]);
title("cos(2*pi*x + 2*pi*y)");
saveas(gcf, "./result/cosine_2d.jpg");
ff = fftshift(fft2(z));
figure;
imagesc(abs(ff));
title("spectrum")
saveas(gcf,"./result/spectrum_cosine.jpg");
figure;
imshow(angle(ff));
title("angle");
saveas(gcf,"./result/angle_cosine.jpg");

%%
figure;
mesh(x,y,abs(ff));
title("spectrum in 3d");
view(20,20);
axis off;
saveas(gcf, "1p_spectrum_3d.jpg");

%%
img = imread('brick.jpg');
img = im2double(rgb2gray(img));
figure;
imshow(img);
title('original');
saveas(gcf, './result/p2_brick.jpg');


ff = fftshift(fft2(img));
figure;
imshow(log(abs(ff)+1), []);
title('spectrum after log transformation');
saveas(gcf, './result/spectrum_brick.jpg');
% imshow(log(abs(ff)),[]);
figure;
imshow(angle(ff));
title('phase angle');
saveas(gcf, './result/phase_angle_brick.jpg');

%%
[m,n] = size(img);
ff(m/2+1,n/2+1) = 0;
img2 = abs(ifft2(ff));
imshow(img2);
title('Average intensity of 0');
saveas(gcf,'./result/avg_intense_0_brick.jpg');



