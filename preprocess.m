function [new_image] = preprocess(image_path)
    old_image = imread(image_path);
    x = imresize(old_image,[256,256]);
    x = rgb2gray(x);
    x = x+50;
    x = medfilt2(x);
    x = imgaussfilt(x)
    x = fibermetric(x);
    new_image = x;
end