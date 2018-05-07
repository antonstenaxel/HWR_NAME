BW = imread('seg (3).png');

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
BW(CC.PixelIdxList{indexOfMax}) = 0;
biggest(CC.PixelIdxList{indexOfMax}) = 1;
 
figure
imshow(BW);

figure
imshow(biggest);

% BW = im2double(BW);
% 
% Z = imadd(BW,biggest);
% 
% figure
% imshow(Z);

% saveas(biggest, 'area.png')

