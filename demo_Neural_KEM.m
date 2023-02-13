% This is a MATLAB demo showing how to use the Neural KEM method for dynamic 
% PET image reconstruction (frame by frame)

% The method is described:

%           "Neural KEM: A Kernel Method with Deep Coefficient Prior for PET Image Reconstruction", 
%           IEEE Transactions on Medical Imaging, in press, Oct. 2022. in press, May 2022 (doi: 10.1109/TMI.2022.3176002).

% Programe Author: Siqi Li and Guobao Wang 
% Last date: 1/12/2023

% The neural KEM reconstuction consists of two separate steps: 
% (1) a KEM step for image update from the projection data
% (2) a deep-learning step in the image domain for updating the kernel coefficient image


clear; clc;

% add path
run('../reconstruction_3.30/fessler/irt/setup'); % your path 
run('../reconstruction_3.30/KER_v0.11/setup');% path
addpath('Rec_functions');
addpath('functions');

% image size
imgsiz = [111 111]; 

%% system matrix
sysflag = 1;  % 1: using Fessler's IRT; 
              % 0: using your own system matrix G 
disp('--- Generating system matrix G ...')
if sysflag
    % require Fessler's IRT matlab toolbox to generate a system matrix
    ig = image_geom('nx', imgsiz(1), 'ny', imgsiz(2), 'fov', 33.3);

    % field of view
    ig.mask = ig.circ(16, 16) > 0;

    % system matrix G
    prjsiz = [249 210];
    sg = sino_geom('par', 'nb', prjsiz(1), 'na', prjsiz(2), 'dr', 70/prjsiz(1), 'strip_width', 'dr');
    G  = Gtomo2_strip(sg, ig, 'single', 1);
    Gopt.mtype  = 'fessler';
    Gopt.ig     = ig;
    Gopt.imgsiz = imgsiz;
else
    Gfold = ''; % where you store your system matrix G;
    load([Gfold, 'G']);	% load matrix G
    Gopt.mtype  = 'matlab';
    Gopt.imgsiz = imgsiz;
end
Gopt.disp = 0; % no display of iterations
Gopt.imgsiz_trunc = [104,104]; % to fit for the deep learning input
Gopt.trunc_range = {4:107, 4:107}; % truncate from 111 x 111 to 104 x 104.

%% Simulate dynamic PET sinograms
disp('--- Generating dynamic images ...')

% phantom
load('zubal_data')

% frame durations
dt = (scant(:,2)-scant(:,1))/60;
numfrm = length(dt);

% dynamic images of activity
X0 = zeros(prod(size(model)),numfrm);
ID = [0 32 64 128 196 255];
for m = 1:numfrm
    for n = 1:length(ID)
        X0(model==ID(n),m) = TAC(m,n)*dt(m);
    end
end

% noise-free geometric projection
for m = 1:size(X0,2)
    proj(:,m) = proj_forw(G, Gopt, X0(:,m));
end

% attenuation factor
ai = exp(-repmat(proj_forw(G, Gopt, u(:)), [1 numfrm]));  

% background (randoms and scatters)
ri = repmat(mean(ai.*proj,1)*0.2,[size(proj,1) 1]);  % 20% uniform background

% total noiseless projection
y0 = ai.*proj + ri; 

% count level
count = 8e6; % a total of 8 million events

% normalized sinograms
cs = count / sum(y0(:));
y0 = cs * y0;
ri = cs * ri;
load ('realizations/y1'); % load a noisy projection or
% yi = poissrnd(y0); % noisy projection
ni = ai*cs; % multiplicative factor

% initial estimate
xinit = [];

%% Build image prior for kernel matrix and neural network input
disp('---Building kernel matrix---')
M = {[1:16],[17:20], [21:24]};
for i = 1:length(M)
        y_i = sum(yi(:,M{i}),2);
        n_i = sum(ni(:,M{i}),2);
        r_i = sum(ri(:,M{i}),2);
        [x, out] = eml_kem(y_i, n_i, G, Gopt, xinit, r_i, 200); 
        x = gaussfilt(x,imgsiz,3);
        U(:,i) = x(:) / sum(dt(M{i}));
end
U = U * diag(1./std(U,1)); % normalization
% Prior image is used for kernel construction and as the input of the neural network.
Prior  = reshape(U,[111,111,3]);
Prior = Prior(Gopt.trunc_range{1}, Gopt.trunc_range{2}, :);
dump(Prior,sprintf('Prior.img'))

%% build the kernel matrix K using k nearest neighbors
sigma = 1;
[N, W] = buildKernel(imgsiz, 'knn', 48, U, 'radial', sigma);
K = buildSparseK(N, W);

%% otherwise you can load an improved kernel matraix (e.g., from deep kernel)
% deep kernel paper and code are available on:https://github.com/SiqiLi1020/Deep-Kernel-Method-for-PET-Image-Reconstruction
% load ('D:/Siqi_work/Deep kernel/K_s/K_64.mat');
% K = K_64;

%% Setting
%select frames to reconstruct 1:24
mm = 24; 
%Outer renconstuction epoch
maxit = 60; 
% sub_iteration number for neural-network learning
sub_iteration_number_frame = 151;
% save the sub_iteration number to fed into the Pytorch code
save('trained/sub_iteration_number_frame.mat','sub_iteration_number_frame');
% note: sub_iteration number for EM updating, default is 1; Other choice as the
% empirical optimiztion if you want a faster convergence rate (such as 10)
sub_EM = 1; 
tic;
for j = mm
    [z, out, Phi] = eml_knem(yi(:,j), ni(:,j), G, Gopt, xinit, ri(:,j), maxit, j, sub_iteration_number_frame, sub_EM, K); 
end
toc;
MSE = 10*log10(sum((z-X0(:,j)).^2)/sum(X0(:,j).^2));
figure,imagesc(reshape(z,imgsiz),[0 270]); axis image; set(gca,'FontSize',16); axis off; colormap(hot);colorbar;
title(sprintf('MSE=%.2fdB', MSE));


