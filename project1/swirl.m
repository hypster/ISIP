close all; clear all;
[im,map] = imread('c_akai.png');
img = ind2rgb(im,map);
figure;
imshow(img)

img=im2double(img);
[m,n,c]=size(img);

midx = m/2;
midy = n/2;
			
swirldegree = 40/1000;	%0.04 choose the nominator between 1 and 100
rot=100;
img2=zeros(size(img));

for i=1:m
    for j=1:n
        yoffset = i - midy;
        xoffset = j - midx;
        [theta, radius] = cart2pol(xoffset, yoffset);
%         alternative way to caculate if you don't want to use standard function cart2pol
%         radian = atan2(yoffset,xoffset); 
%         radius = sqrt(xoffset*xoffset+yoffset*yoffset);
        
%         xx = round(radius*cos(theta)+midx);
%         yy = round(radius*sin(theta)+midy);
%         if you leave radius*degree part then the result is the same image
        xx = round(radius*cos(theta+radius*swirldegree)+midx);
        yy = round(radius*sin(theta+radius*swirldegree)+midy);

        if yy>=1 && yy<=m && xx>=1 && xx<=n
            img2(i,j,:)=img(yy,xx,:);
        end
    end
end
    
figure;
imshow(img2);





