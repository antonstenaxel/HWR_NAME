BW = imread('otsu-2.png');
gray = rgb2gray(BW);
BW = gray>100;
figure
imshow(BW)

% CC = bwconncomp(BW)
% 
% numPixels = cellfun(@numel,CC.PixelIdxList);
% [unused,idx] = max(numPixels);
% BW(CC.PixelIdxList{idx}) = 0;
% 
% figure
% imshow(BW)

CC = bwconncomp(BW);
numOfPixels = cellfun(@numel,CC.PixelIdxList);
[unused,indexOfMax] = max(numOfPixels);
biggest = zeros(size(BW));
biggest(CC.PixelIdxList{indexOfMax}) = 1;

figure
imshow(biggest);

