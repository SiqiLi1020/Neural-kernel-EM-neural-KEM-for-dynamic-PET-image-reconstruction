function [z, out, Phi] = eml_knem(yi, ni, G, Gopt, x0, ri, maxit, m, sub_iter, sub_EM, K)
%--------------------------------------------------------------------------
% Neural KEM main function for PET image reconstruction. 
%
%
%--------------------------------------------------------------------------
% INPUT:
%   yi          sinogram in vector
%   ni          multiplicative factor (normalization, attenuation), can be empty
%   G           system matrix
%   Gopt        option set for G, can be empty if G is a matlab sparse matrix
%   ri          addtive factor (randoms and scatters), can be empty
%   x0          initial image estimate, can be empty
%   maxit       maximum iteration number
%   m           Reconstructed frame
%   sub_iter    subiterations for neural-network learning
%   sub_EM      subiterations for EM update
%   K           kernel matrix accounting for image piror

%
% OUTPUT
%   z       image estimate in vector
%   out     output
%   Phi     objective function value
%
%--------------------------------------------------------------------------
% Programmer: Siqi Li @ UC Davis, Wang Lab, 12/29/2021
% 
%--------------------------------------------------------------------------
%% check inputs
imgsiz = Gopt.imgsiz;
numpix = prod(imgsiz);
if isempty(x0)
    x0 = ones(numpix,1);
end
[yi, ri, ni] = sino_preprocess(yi, ri, ni);

if isempty(sub_iter)
    sub_iter = 151;
end

if isempty(sub_EM)
    sub_EM = 1;
end

if nargin<11
    K = speye(numpix);
end
ktype = 'org';

% set Gopt
Gopt = setGopt(ni, G, Gopt);
Gopt.sens = ker_back(K,Gopt.sens,ktype); % MUST
if isempty(maxit)
    maxit = 10;
end

% initialization
x    = max(mean(x0(:))*1e-9,x0(:)); z(~Gopt.mask) = 0;
z    = K * x;
yeps = mean(yi(:))*1e-9;
wx   = Gopt.sens;

% output
if nargin>1
    out = []; Phi = [];
end
out.xest = zeros(length(x(:)), min(maxit,ceil(maxit/Gopt.savestep)+1));
t1 = cputime;

Gopt.savestep = 1;
%% iterative loop
for it = 1:maxit     
    
    disp(sprintf('frame %d Iteration %d', m, it));
    % save data
    if Gopt.disp
        disp(sprintf('iteration %d',it));
    end
    if nargout>1 & ( it==1 | rem(it,Gopt.savestep)==0 )
        itt = min(it, floor(it/Gopt.savestep) + 1);
        out.step(itt)   = it;
        out.time(:,itt) = cputime - t1;        
        out.xest(:,itt) = x;
        
    end
    
    % EM update
    for ii = 1:sub_EM
        z  = ker_forw(K,x,ktype);
        out.zest(:,it) = z;
        yb = ni.*proj_forw(G, Gopt, z) + ri;
        yy = yi./(yb+yeps);
        yy(yb==0&yi==0) = 1;
        zb = proj_back(G, Gopt, ni.*yy);
        xb = ker_back(K,zb,ktype);
        x  = x ./ wx .* xb;
        x(~Gopt.mask) = 0;
    end
    % Initialization model
    if it == 1
        sub_iteration_frame = 0;
        save('trained/sub_iteration_frame.mat','sub_iteration_frame');
    end
    % save label image and weight image
    temp = x;
    weight_img = wx;
    temp = reshape(temp,Gopt.imgsiz(1),Gopt.imgsiz(2));
    weight_img = reshape(weight_img, Gopt.imgsiz(1),Gopt.imgsiz(2));
    u_int = temp(Gopt.trunc_range{1}, Gopt.trunc_range{2});
    weight_int = weight_img(Gopt.trunc_range{1}, Gopt.trunc_range{2});
    dump(u_int,sprintf('noise_input_2D.img'));
    dump(weight_int,sprintf('weight_img_2D.img'));
    % neural-network learning
    system('python DIP_OT.py');
    result = touch('DIP_output_2D.img');
    result = reshape(result,Gopt.imgsiz_trunc(1), Gopt.imgsiz_trunc(2));
    DIP_out = zeros(Gopt.imgsiz(1), Gopt.imgsiz(2));
    DIP_out(Gopt.trunc_range{1}, Gopt.trunc_range{2}) = result;
    x = DIP_out(:);
    sub_iteration_frame = sub_iter;
    save('trained/sub_iteration_frame.mat','sub_iteration_frame');
    if it == maxit
       z  = ker_forw(K,x,ktype);
       out.zest(:,it+1) = z(:);
    end
     
end
% last update
z  = ker_forw(K,x,ktype);

proj_clear(Gopt);

% ------------------------------------------------------------------------
% kernel forward projection
% ------------------------------------------------------------------------
function y = ker_forw(K, x, type)
switch type
    case 'org'
        y = K * x;
    case 'psd'
        y = K'*(K*x);
end

% ------------------------------------------------------------------------
% kernel back projection
% ------------------------------------------------------------------------
function x = ker_back(K, y, type)
switch type
    case 'org'
        x = K' * y;
    case 'psd'
        x = K'*(K*y);
end


    
