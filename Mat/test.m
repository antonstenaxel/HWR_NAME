BW = imread('worst_bin.jpg');

% Converting into logical image
gray = rgb2gray(BW);
BW = gray>80;

% Taking the bigest connected component
BW2 = bwareafilt(BW,1);

% creating a mask from the binary image
mask = imfill (BW2, 'holes');

figure
imshow(mask)

% Inew = mask.*BW;
% imwrite(Inew,'best_seg.jpg')

IG = imread('worst_orig.jpg');

% gray = rgb2gray(IG);
% IG = gray>100;

mask = im2uint8(mask);
% mask = double(mask);
% IG = double(IG);

figure
imshow(IG)

Inew = mask.*IG;
% maskedRgbImage = bsxfun(@times, IG, cast(mask, 'like', IG));

figure
imshow(Inew)

imwrite(Inew,'worst_seg.jpg')
