% author : maplect
% Copyright (c) [2020] [maplect]
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
function vec = get_hu_vec(im)
if ndims(im) == 3
    im = rgb2gray(im);
end
im = double(im);
m00 = sum(sum(im));
m10 = 0;
m01 = 0;
[row,col] = size(im);
for i = 1:row
    for j = 1:col
        m10 = m10+i*im(i,j);
        m01 = m01+j*im(i,j);
    end
end
u10 = m10/m00;
u01 = m01/m00;
m20 = 0;
m02 = 0;
m11 = 0;
m30 = 0;
m12 = 0;
m21 = 0;
m03 = 0;
for i = 1:row
    for j = 1:col
        m20 = m20+i^2*im(i,j);
        m02 = m02+j^2*im(i,j);
        m11 = m11+i*j*im(i,j);
        m30 = m30+i^3*im(i,j);
        m03 = m03+j^3*im(i,j);
        m12 = m12+i*j^2*im(i,j);
        m21 = m21+i^2*j*im(i,j);
    end
end
y11 = m11-u01*m10;
y20 = m20-u10*m10;
y02 = m02-u01*m01;
y30 = m30-3*u10*m20+2*u10^2*m10;
y12 = m12-2*u01*m11-u10*m02+2*u01^2*m10;
y21 = m21-2*u10*m11-u01*m20+2*u10^2*m01;
y03 = m03-3*u01*m02+2*u01^2*m01;
n20 = y20/m00^2;
n02 = y02/m00^2;
n11 = y11/m00^2;
n30 = y30/m00^2.5;
n03 = y03/m00^2.5;
n12 = y12/m00^2.5;
n21 = y21/m00^2.5;
h1 = n20 + n02;
h2 = (n20-n02)^2 + 4*(n11)^2;
h3 = (n30-3*n12)^2 + (3*n21-n03)^2;
h4 = (n30+n12)^2 + (n21+n03)^2;
h5 = (n30-3*n12)*(n30+n12)*((n30+n12)^2-3*(n21+n03)^2)+(3*n21-n03)*(n21+n03)*(3*(n30+n12)^2-(n21+n03)^2);
h6 = (n20-n02)*((n30+n12)^2-(n21+n03)^2)+4*n11*(n30+n12)*(n21+n03);
h7 = (3*n21-n03)*(n30+n12)*((n30+n12)^2-3*(n21+n03)^2)+(3*n12-n30)*(n21+n03)*(3*(n30+n12)^2-(n21+n03)^2);
vec = [h1 h2 h3 h4 h5 h6 h7];
vec = vec ./ sum(vec);