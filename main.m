close all;
clear all;
dog_folder_path='C:\Users\Azra\pet-recognition-master\pet-recognition-master\training\dog';
non_dog_folder_path='C:\Users\Azra\pet-recognition-master\pet-recognition-master\training\cat';

[dog_wave,imgDog] = dc_wavelet(dog_folder_path);
[non_dog_wave,imgNonDog] = dc_wavelet(non_dog_folder_path);

feature = 25; % 1 < feature < 80

[result,w,U,S,V,threshold] = dc_trainer(dog_wave,non_dog_wave,feature);

% Test on the testing dataset:
TestSet='./testing'
[Test_wave,imgTest] = dc_wavelet(TestSet); % wavelet transformation
TestMat = U'*Test_wave; % SVD projection
pval = w'*TestMat; % LDA projection
hiddenlabels=[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
TestNum = length(pval);
%nonDog = 1, dog = 0
ResVec = (pval>threshold)
disp('Number of mistakes');
errNum = sum(abs(ResVec - hiddenlabels))
disp('Rate of success');
sucRate = 1-errNum/TestNum
k = 1;
figure(2)
for i = 1:TestNum
    if ResVec(i)~=hiddenlabels(i)
        S = imgTest{i};
        subplot(2,2,k)
        imshow(uint8(S))
        if k<4
            k = k+1;
        else
            k=1;
        end
    end
end


