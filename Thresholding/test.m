BW = imread('4.jpg');

% Converting into logical image
gray = rgb2gray(BW);
BW = gray>80;

% figure
% imshow(gray)

% Taking the bigest connected component
BW2 = bwareafilt(BW,1);

% creating a mask from the binary image
mask = imfill (BW2, 'holes');

Inew = mask.*BW;

figure
imshow(Inew)

imwrite(Inew,'3_seg.jpg')

% IG = imread('3_gray.jpg');

% gray = rgb2gray(IG);
% IG = gray>100;

% mask = im2uint8(mask);
% mask = double(mask);
% IG = double(IG);

% Inew = mask.*repmat(IG,[1,1,255]);
% maskedRgbImage = bsxfun(@times, IG, cast(mask, 'like', IG));
