close all; clear all; 
% src_dir = '/Users/zhangrao/Documents/NOW/EE368/hw/hw4/hw4_data/hw4_radiograph_';
file_no = 2;
graph = im2double(imread('hw4_radiograph_2.jpg'));
figure;
imshow(graph);
imwrite(graph, strcat('q1_', num2str(file_no), '.png'));

% part (a)
N = 10;
med_graph = medfilt2(graph, [N N]);
figure;
imshow(med_graph);
imwrite(med_graph, strcat('q1_med_', num2str(file_no), '.png'));

% part (b)
fft_graph = fftshift(fft2(graph));
abs_graph = abs(fft_graph);
log_graph = log(abs_graph);
norm_graph = log_graph / max(max(log_graph));
figure; 
imshow(norm_graph);
imwrite(norm_graph, strcat('q1_fft_', num2str(file_no), '.png'));

% part (c)
if file_no == 1
    freqs = [270 80; 260 125; 244 215; 240 260];
    sigma = [20 20 20 20];
else
    freqs = [275 445; 210 450];
    sigma = [10 10];
end

[rows, cols] = size(graph);
[omega_x, omega_y] = meshgrid(1:cols, 1:rows);  % x represents n_col, y represents n_row
[n_freqs, ~] = size(freqs);
notch_filter = ones(rows, cols);

for ii = 1 : n_freqs
    temp_square = (omega_x - freqs(ii, 1)) .^ 2 + (omega_y - freqs(ii, 2)) .^ 2;
    notch_filter = notch_filter .* (1 - exp(-temp_square ./ (sigma(ii) .^ 2)));
end
notch_fft_graph = fft_graph .* notch_filter;
abs_graph = abs(notch_fft_graph);
log_graph = log(abs_graph);
norm_graph = log_graph / max(max(log_graph));
figure;
imshow(norm_graph);
imwrite(norm_graph, strcat('q1_filter_freq_', num2str(file_no), '.png'));

ifft_graph = ifft2(ifftshift(notch_fft_graph));
real_graph = real(ifft_graph);
figure;
imshow(real_graph);
imwrite(real_graph, strcat('q1_filter_real_', num2str(file_no), '.png'));
