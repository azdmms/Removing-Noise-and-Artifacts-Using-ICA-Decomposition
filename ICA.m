%%
clc
clear all
%% part 1
%% 
load 'C:\Users\DearUser\Downloads\Lab 3\data\mecg1.dat' ;
load 'C:\Users\DearUser\Downloads\Lab 3\data\fecg1.dat' ;
load 'C:\Users\DearUser\Downloads\Lab 3\data\noise1.dat' ;
sigtot=fecg1+mecg1+noise1;
%% problem 1
freq = 256 ;
nbsamples = 2560 ;
time = (0:1/freq:(nbsamples-1)/freq) ;
%total
figure ;
subplot(4 , 1 , 1) ;
set(gcf, 'WindowState', 'maximized');
plot(time , sigtot) ;
xlim([0 , (nbsamples-1)/freq]) ;
title('EEG Signal total(mV)') ;
xlabel('time (s)') ;
ylabel('EEG Signal') ;
%janin
%figure ;
subplot(4 , 1 , 2) ;
set(gcf, 'WindowState', 'maximized');
plot(time , fecg1) ;
xlim([0 , (nbsamples-1)/freq]) ;
title('EEG Signal janin(mV)') ;
xlabel('time (s)') ;
ylabel('EEG Signal') ;
%mother
%figure ;
subplot(4 , 1 , 3) ;
set(gcf, 'WindowState', 'maximized');
plot(time , mecg1) ;
xlim([0 , (nbsamples-1)/freq]) ;
title('EEG Signal mother(mV)') ;
xlabel('time (s)') ;
ylabel('EEG Signal') ;

%figure ;
subplot(4 , 1 , 4) ;
set(gcf, 'WindowState', 'maximized');
plot(time , noise1) ;
xlim([0 , (nbsamples-1)/freq]) ;
title('EEG Signal noise(mV)') ;
xlabel('time (s)') ;
ylabel('EEG Signal') ;
%% problem2 
%total
figure;
subplot(4 , 1 , 1) ;
pwelch(sigtot) ;
title('sig  total') ; 
%mother
subplot(4 , 1 , 2) ;
pwelch(mecg1) ;
title('mother ') ; 
%janin
subplot(4 , 1 , 3) ;
pwelch(fecg1) ;
title('janin ') ;
%noise
subplot(4 , 1 , 4) ;
pwelch(noise1) ;
title('noise ') ;
%% problem 3
%signal total
meantot=mean(sigtot(:,1));
variancetot=var(sigtot(:,1));
%mother
meanmother=mean(mecg1(:,1));
variancemother=var(mecg1(:,1));
%child
meanchild=mean(fecg1(:,1));
variancechild=var(fecg1(:,1));
%noise
meannoise=mean(noise1(:,1));
variancenoise=var(noise1(:,1));
%% problem 4
%% histogram
nbins = 150;
%signal total
subplot(4 , 1 , 1) ;
hist(sigtot(:,1),nbins);
title('EEG Signal total') ;
%signal mother
subplot(4 , 1 , 2) ;
hist(mecg1(:,1),nbins);
title('EEG Signal mother') ;
%signal child
subplot(4 , 1 , 3) ;
hist(fecg1(:,1),nbins);
title('EEG Signal child') ;
%signal mother
subplot(4 , 1 , 4) ;
hist(noise1(:,1),nbins);
title('EEG Signal noise') ;
%% momentum
%signal total
momtotal=kurtosis(sigtot);
%signal mother
mommother=kurtosis(mecg1);
%signal child
momchild=kurtosis(fecg1);
%signal noise
momnoise=kurtosis(noise1);
%% part 2 
clear all;
clc;
%% problem 1
Fs=256;
plot_title='data'
load 'C:\Users\DearUser\Downloads\Lab 3\data\X.dat' ;
plot3ch(X,Fs,plot_title)
[U,S,V]=svd(X);
%% problem 2
for i=1:1:3
plot3dv(V(:,i), S(:,i), 'k');
end
%% problem 3
freq = 256 ;
nbsamples = 2560 ;
time = (0:1/freq:(nbsamples-1)/freq) ;

subplot(3 , 1 , 1) ;
plot(time,U(:,1));
title('1st column') ;

subplot(3 , 1 , 2) ;
plot(time,U(:,2));
title('2nd column') ;

subplot(3 , 1 , 3) ;
plot(time,U(:,3));
title('3rd column') ;
figure;
subplot(3 , 1 , 1) ;
stem(U(:,1));
xlim([0 , 2560]) ;
title('1st column') ;
subplot(3 , 1 , 2) ;
stem(U(:,2));
xlim([0 , 2560]) ;
title('2nd column') ;
subplot(3 , 1 , 3) ;
stem(U(:,3));
xlim([0 , 2560]) ;
title('3rd column') ;
%% problem 4
S_new = S ;
bad = [1 , 3] ;
good = [2] ;
for i = bad
    S_new(: , i) = 0 ;
end
for i = good
    S_new(: , i) = S(: , i) ;
end
X_new=U*S_new*V';
subplot(3 , 1 , 1) ;
plot(time,X_new(:,1));
title('1st column new') ;

subplot(3 , 1 , 2) ;
plot(time,X_new(:,2));
title('2nd column new') ;

subplot(3 , 1 , 3) ;
plot(time,X_new(:,3));
title('3rd column new') ;

%% part 3
clear all;
clc;
%% problem 1
Fs=256;
load 'C:\Users\DearUser\Downloads\Lab 3\data\X.dat' ;
X_inverse=X';
[W, Zhat] = ica(X_inverse);
A_estimate=inv(W);

%% problem 2
Fs=256;
plot_title='data';
plot3ch(X,Fs,plot_title)
%%
for i= 1:1:3
plot3dv(A_estimate(:,i), Zhat(:,i), 'k')
end

%% problem 3
freq = 256 ;
nbsamples = 2560 ;
time = (0:1/freq:(nbsamples-1)/freq) ;
figure;
subplot(3 , 1 , 1) ;
plot(time,Zhat(1,:));
title('1st row') ;
subplot(3 , 1 , 2) ;
plot(time,Zhat(2,:));
title('2nd row') ;
subplot(3 , 1 , 3) ;
plot(time,Zhat(3,:));
title('3rd row') ;
%%
A_estimate_new = A_estimate ;
bad = [1 , 2] ;
good = [3] ;
for i = bad
    A_estimate_new(: , i) = 0 ;
end
for i = good
    A_estimate_new(: , i) = A_estimate(: , i) ;
end
X_new_ica=A_estimate_new*Zhat;
%% problem 4
subplot(3 , 1 , 1) ;
plot(time,X_new_ica(1,:));
title('1st column new') ;

subplot(3 , 1 , 2) ;
plot(time,X_new_ica(2,:));
title('2nd column new') ;

subplot(3 , 1 , 3) ;
plot(time,X_new_ica(3,:));
title('3rd column new') ;

%% part 4
subplot(3 , 2 , 1) ;
scatter3(X(:,1),X(:,2),X(:,3));
ylabel('2 channel') ;
xlabel('1 channel ') ;
zlabel('3 channel ') ;
title('scatter X') ;
%figure;
subplot(3 , 2 , 2) ;
scatter3(X_new(:,1),X_new(:,2),X_new(:,3));
ylabel('2 channel') ;
xlabel('1 channel ') ;
zlabel('3 channel ') ;
title('scatter X SVD') ;

%figure;
subplot(3 , 2 , 3) ;
X_new_ica1=X_new_ica';
scatter3(X_new_ica1(:,1),X_new_ica1(:,2),X_new_ica1(:,3));
ylabel('2 channel') ;
xlabel('1 channel ') ;
zlabel('3 channel ') ;
title('scatter X ICA') ;

subplot(3 , 2 , 4) ;
for i=1:1:3
plot3dv(V(:,i), S(:,i), 'k');
end
ylabel('2 channel') ;
xlabel('1 channel ') ;
zlabel('3 channel ') ;
title('column SVD') ;
S=S';
for i= 1:1:3
[V1,V2,V3]=plot3dv1(V(:,i), S(:,i), 'k');
end
Theta1 = atan2(norm(cross(V1, V2)), dot(V1, V2));
Theta2 = atan2(norm(cross(V1, V3)), dot(V1, V3));
Theta3 = atan2(norm(cross(V3, V2)), dot(V3, V2));

subplot(3 , 2 , 5) ;
for i= 1:1:3
plot3dv(A_estimate(:,i), Zhat(:,i), 'k')
end
ylabel('2 channel') ;
xlabel('1 channel ') ;
zlabel('3 channel ') ;
title('column ICA') ;
for i= 1:1:3
[W1,W2,W3]=plot3dv1(A_estimate(:,i), Zhat(:,i), 'k');
end
Theta11 = atan2(norm(cross(W1, W2)), dot(W1, W2));
Theta22 = atan2(norm(cross(W1, W3)), dot(W1, W3));
Theta33 = atan2(norm(cross(W3, W2)), dot(W3, W2));
%% problem 2
load 'C:\Users\DearUser\Downloads\Lab 3\data\fecg2.dat' ;

subplot(3 , 1 , 1) ;
plot(fecg2);
xlim([0 , 2560]) ;
title('ideal signal') ;
subplot(3 , 1 , 2) ;
plot(X_new_ica(1,:));
xlim([0 , 2560]) ;
title('ICA signal') ;
subplot(3 , 1 , 3) ;
plot(X_new(:,1));
xlim([0 , 2560]) ;
title('SVD signal') ;
%% problem 3
R_ICA = corrcoef(fecg2,X_new_ica(1,:));
R_SVD = corrcoef(fecg2,X_new(:,1));
%% function declaration

function plot3ch(X,Fs,plot_title)
%PLOT3CH  Plot 3 channel data in the time-domain and on a 3D scatter plot
%  PLOT3CH(X,FS,'TITLE') plots the three columns of data matrix X on a
%  time-domain plot with sample rate FS on and plots each column against the
%  other on a 3D scatter plot. The default value for FS is 256 Hz. The optional
%  'TITLE' input allows the user to specify the plot title string.
% Created by: G.D. Clifford 2004 gari AT mit DOT edu
% Modified 5/6/05, Eric Weiss. Documentation updated. Plot title input added.
% Input argument checking
%------------------------
if nargin < 2
    Fs = 256;
end;
if nargin < 3
    plot_title = '3 Channel Data';
end;
[M,N] = size(X);
if N ~= 3;
    error('Input matrix must have 3 columns!');
end;
% Generate time-domain plot
%--------------------------
t = [1/Fs:1/Fs:M/Fs];
figure;
for i = 1:N
    subplot(N,1,i)
    plot(t,X(:,i)); ylabel(['Ch',int2str(i)]);
    axis([0 max(t) min(X(:,i))-abs(0.1*max(X(:,i))) max(X(:,i))+abs(0.1*max(X(:,i)))]);
    %axis([0 max(t) min(X(:,i)) max(X(:,i))])
end;
xlabel('Time (sec)');
subplot(N,1,1); title(plot_title);
figure;
plot3(X(:,1), X(:,2), X(:,3),'.m');
xlabel('Ch1'); ylabel('Ch2'); zlabel('Ch3');
title(plot_title);
grid on;
end




function plot3dv(v, s, col)
%PLOT3DV  Plots the specified vector onto a 3D scatter plot
%  PLOT3DV(V, S, 'COL') plots the eigenvector +/-V with singular value S and
%  color 'COL' onto a 3D plot of the currently displayed figure. The length of
%  the plotted eigenvector is equal to the square root of the singular value. If
%  the singular value S is not specified, the default scaling length is 10. If
%  the color 'COL' is not specified, the default color is 'k' (black).

% Created by: GD Clifford 2004 gari AT alum DOT mit DOT edu
% Last modified 5/7/06, Eric Weiss. Documentation updated.

% Input argument checking
%------------------------
if nargin < 2 | isempty(s)
    s = 100;
end;
if nargin < 3
    col = 'k';
end;
v = v(:); % ensure that eigenvector is in column format
[m, n] = size(v);
if (n ~= 1 | m ~= 3)
    error('vector must be 3x1')
end;
if s == 1  % legacy code: does not affect function
    ln = 1/sqrt((v(1)*v(1))+(v(2)*v(2))+(v(3)*v(3)));
end;

% Plot eigenvector on 3D plot
%----------------------------
sn = sqrt(s);
hold on;
plot3(sn*[-1*v(1) v(1)],sn*[-1*v(2) v(2)],sn*[-1*v(3) v(3)],col);
grid on;
view([1,1,1])
end


function [V1,V2,V3] =plot3dv1(v, s, col)
%PLOT3DV  Plots the specified vector onto a 3D scatter plot
%  PLOT3DV(V, S, 'COL') plots the eigenvector +/-V with singular value S and
%  color 'COL' onto a 3D plot of the currently displayed figure. The length of
%  the plotted eigenvector is equal to the square root of the singular value. If
%  the singular value S is not specified, the default scaling length is 10. If
%  the color 'COL' is not specified, the default color is 'k' (black).

% Created by: GD Clifford 2004 gari AT alum DOT mit DOT edu
% Last modified 5/7/06, Eric Weiss. Documentation updated.

% Input argument checking
%------------------------
if nargin < 2 | isempty(s)
    s = 100;
end;
if nargin < 3
    col = 'k';
end;
v = v(:); % ensure that eigenvector is in column format
[m, n] = size(v);
if (n ~= 1 | m ~= 3)
    error('vector must be 3x1')
end;
if s == 1  % legacy code: does not affect function
    ln = 1/sqrt((v(1)*v(1))+(v(2)*v(2))+(v(3)*v(3)));
end;

% Plot eigenvector on 3D plot
%----------------------------
sn = sqrt(s);
hold on;
V1=sn*[-1*v(1) v(1)]
V2=sn*[-1*v(2) v(2)]
V3=sn*[-1*v(3) v(3)];

end


function [W, Zhat] = ica(X)
%ICA  Perform independent component analysis
%  [W, ZHAT] = ICA(X) performs independent component analysis on data
%  observation matrix X. Matrix X is a transposed observation matrix, such that
%  each row of X represents an observed signal. This approach uses Cardoso's ICA
%  algorithm to estimate sources (ZHAT) and the de-mixing matrix W, an
%  approximation to A^{-1}, the original (unknown) mixing matrix. 

% Created by: G D Clifford 2004  gari AT mit DOT edu
% Last Modified: 5/7/06, documentation updated.

% Input argument checking
%------------------------
[a, b] = size(X);
if a > b
    fprintf('Warning - ICA works across the rows of the input data.\n');
    error('Please transpose input.');
end
Nsources = a;

if Nsources > min([a b])
    Nsources = min([a b]);
    fprintf('Warning - number of soures cannot exceed number of observation channels \n')
    fprintf(' ... reducing to %i \n',Nsources)
end

%tic
[Winv, Zhat] = jade(X,Nsources);
W = pinv(Winv);
%fprintf('algorithm timing ...  ')
%toc
end


function [A,S]=jade(X,m)
% Source separation of complex signals with JADE.
% Jade performs `Source Separation' in the following sense:
%   X is an n x T data matrix assumed modelled as X = A S + N where
% 
% o A is an unknown n x m matrix with full rank.
% o S is a m x T data matrix (source signals) with the properties
%    	a) for each t, the components of S(:,t) are statistically
%    	   independent
% 	b) for each p, the S(p,:) is the realization of a zero-mean
% 	   `source signal'.
% 	c) At most one of these processes has a vanishing 4th-order
% 	   cumulant.
% o  N is a n x T matrix. It is a realization of a spatially white
%    Gaussian noise, i.e. Cov(X) = sigma*eye(n) with unknown variance
%    sigma.  This is probably better than no modeling at all...
%
% Jade performs source separation via a 
% Joint Approximate Diagonalization of Eigen-matrices.  
%
% THIS VERSION ASSUMES ZERO-MEAN SIGNALS
%
% Input :
%   * X: Each column of X is a sample from the n sensors
%   * m: m is an optional argument for the number of sources.
%     If ommited, JADE assumes as many sources as sensors.
%
% Output :
%    * A is an n x m estimate of the mixing matrix
%    * S is an m x T naive (ie pinv(A)*X)  estimate of the source signals
%
%
% Version 1.5.  Copyright: JF Cardoso.  
%
% See notes, references and revision history at the bottom of this file



[n,T]	= size(X);

%%  source detection not implemented yet !
if nargin==1, m=n ; end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A few parameters that could be adjusted
nem	= m;		% number of eigen-matrices to be diagonalized
seuil	= 1/sqrt(T)/100;% a statistical threshold for stopping joint diag


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% whitening
%
if m<n, %assumes white noise
 	[U,D] 	= eig((X*X')/T); 
	[puiss,k]=sort(diag(D));
 	ibl 	= sqrt(puiss(n-m+1:n)-mean(puiss(1:n-m)));
 	bl 	= ones(m,1) ./ ibl ;
 	W	= diag(bl)*U(1:n,k(n-m+1:n))';
 	IW 	= U(1:n,k(n-m+1:n))*diag(ibl);
else    %assumes no noise
 	IW 	= sqrtm((X*X')/T);
 	W	= inv(IW);
end;
Y	= W*X;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Cumulant estimation


R	= (Y*Y' )/T ;
C	= (Y*Y.')/T ;

Yl	= zeros(1,T);
Ykl	= zeros(1,T);
Yjkl	= zeros(1,T);

Q	= zeros(m*m*m*m,1) ;
index	= 1;

for lx = 1:m ; Yl 	= Y(lx,:);
for kx = 1:m ; Ykl 	= Yl.*conj(Y(kx,:));
for jx = 1:m ; Yjkl	= Ykl.*conj(Y(jx,:));
for ix = 1:m ; 
	Q(index) = ...
	(Yjkl * Y(ix,:).')/T -  R(ix,jx)*R(lx,kx) -  R(ix,kx)*R(lx,jx) -  C(ix,lx)*conj(C(jx,kx))  ;
	index	= index + 1 ;
end ;
end ;
end ;
end

%% If you prefer to use more memory and less CPU, you may prefer this
%% code (due to J. Galy of ENSICA) for the estimation the cumulants
%ones_m = ones(m,1) ; 
%T1 	= kron(ones_m,Y); 
%T2 	= kron(Y,ones_m);  
%TT 	= (T1.* conj(T2)) ;
%TS 	= (T1 * T2.')/T ;
%R 	= (Y*Y')/T  ;
%Q	= (TT*TT')/T - kron(R,ones(m)).*kron(ones(m),conj(R)) - R(:)*R(:)' - TS.*TS' ;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%computation and reshaping of the significant eigen matrices

[U,D]	= eig(reshape(Q,m*m,m*m)); 
[la,K]	= sort(abs(diag(D)));

%% reshaping the most (there are `nem' of them) significant eigenmatrice
M	= zeros(m,nem*m);	% array to hold the significant eigen-matrices
Z	= zeros(m)	; % buffer
h	= m*m;
for u=1:m:nem*m, 
	Z(:) 		= U(:,K(h));
	M(:,u:u+m-1)	= la(h)*Z;
	h		= h-1; 
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% joint approximate diagonalization of the eigen-matrices


%% Better declare the variables used in the loop :
B 	= [ 1 0 0 ; 0 1 1 ; 0 -i i ] ;
Bt	= B' ;
Ip	= zeros(1,nem) ;
Iq	= zeros(1,nem) ;
g	= zeros(3,nem) ;
G	= zeros(2,2) ;
vcp	= zeros(3,3);
D	= zeros(3,3);
la	= zeros(3,1);
K	= zeros(3,3);
angles	= zeros(3,1);
pair	= zeros(1,2);
c	= 0 ;
s	= 0 ;


%init;
encore	= 1;
V	= eye(m); 

% Main loop
while encore, encore=0;
 for p=1:m-1,
  for q=p+1:m,

 	Ip = p:m:nem*m ;
	Iq = q:m:nem*m ;

	% Computing the Givens angles
 	g	= [ M(p,Ip)-M(q,Iq)  ; M(p,Iq) ; M(q,Ip) ] ; 
 	[vcp,D] = eig(real(B*(g*g')*Bt));
	[la, K]	= sort(diag(D));
 	angles	= vcp(:,K(3));
	if angles(1)<0 , angles= -angles ; end ;
 	c	= sqrt(0.5+angles(1)/2);
 	s	= 0.5*(angles(2)-j*angles(3))/c; 

 	if abs(s)>seuil, %%% updates matrices M and V by a Givens rotation
	 	encore 		= 1 ;
		pair 		= [p;q] ;
 		G 		= [ c -conj(s) ; s c ] ;
		V(:,pair) 	= V(:,pair)*G ;
	 	M(pair,:)	= G' * M(pair,:) ;
		M(:,[Ip Iq]) 	= [ c*M(:,Ip)+s*M(:,Iq) -conj(s)*M(:,Ip)+c*M(:,Iq) ] ;
 	end%% if
  end%% q loop
 end%% p loop
end%% while

%%%estimation of the mixing matrix and signal separation
A	= IW*V;
S	= V'*Y ;

return ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Note 1: This version does *not* assume circularly distributed
% signals as 1.1 did.  The difference only entails more computations
% in estimating the cumulants
%
%
% Note 2: This code tries to minimize the work load by jointly
% diagonalizing only the m most significant eigenmatrices of the
% cumulant tensor.  When the model holds, this avoids the
% diagonalization of m^2 matrices.  However, when the model does not
% hold, there is in general more than m significant eigen-matrices.
% In this case, this code still `works' but is no longer equivalent to
% the minimization of a well defined contrast function: this would
% require the diagonalization of *all* the eigen-matrices.  We note
% (see the companion paper) that diagonalizing **all** the
% eigen-matrices is strictly equivalent to diagonalize all the
% `parallel cumulants slices'.  In other words, when the model does
% not hold, it could be a good idea to diagonalize all the parallel
% cumulant slices.  The joint diagonalization will require about m
% times more operations, but on the other hand, computation of the
% eigen-matrices is avoided.  Such an approach makes sense when
% dealing with a relatively small number of sources (say smaller than
% 10).
%
%
% Revision history
%-----------------
%
% Version 1.5 (Nov. 2, 97) : 
% o Added the option kindly provided by Jerome Galy
%   (galy@dirac.ensica.fr) to compute the sample cumulant tensor.
%   This option uses more memory but is faster (a similar piece of
%   code was also passed to me by Sandip Bose).
% o Suppressed the useles variable `oui'.
% o Changed (angles=sign(angles(1))*angles) to (if angles(1)<0 ,
%   angles= -angles ; end ;) as suggested by Iain Collings
%   <i.collings@ee.mu.OZ.AU>.  This is safer (with probability 0 in
%   the case of sample statistics)
% o Cosmetic rewriting of the doc.  Fixed some typos and added new
%   ones.
%
% Version 1.4 (Oct. 9, 97) : Changed the code for estimating
% cumulants. The new version loops thru the sensor indices rather than
% looping thru the time index.  This is much faster for large sample
% sizes.  Also did some clean up.  One can now change the number of
% eigen-matrices to be jointly diagonalized by just changing the
% variable `nem'.  It is still hard coded below to be equal to the
% number of sources.  This is more economical and OK when the model
% holds but not appropriate when the model does not hold (in which
% case, the algorithm is no longer asymptotically equivalent to
% minimizing a contrast function, unless nem is the square of the
% number of sources.)
%
% Version 1.3 (Oct. 6, 97) : Added various Matalb tricks to speed up
% things a bit.  This is not very rewarding though, because the main
% computational burden still is in estimating the 4th-order moments.
% 
% Version 1.2 (Mar., Apr., Sept. 97) : Corrected some mistakes **in
% the comments !!**, Added note 2 `When the model does not hold' and
% the EUSIPCO reference.
%
% Version 1.1 (Feb. 94): Creation
%
%-------------------------------------------------------------------
%
% Contact JF Cardoso for any comment bug report,and UPDATED VERSIONS.
% email : cardoso@sig.enst.fr 
% or check the WEB page http://sig.enst.fr/~cardoso/stuff.html 
%
% Reference:
%  @article{CS_iee_94,
%   author = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",
%   journal = "IEE Proceedings-F",
%   title = "Blind beamforming for non {G}aussian signals",
%   number = "6",
%   volume = "140",
%   month = dec,
%   pages = {362-370},
%   year = "1993"}
%
%
%  Some analytical insights into the asymptotic performance of JADE are in
% @inproceedings{PerfEusipco94,
%  HTML 	= "ftp://sig.enst.fr/pub/jfc/Papers/eusipco94_perf.ps.gz",
%  author       = "Jean-Fran\c{c}ois Cardoso",
%  address      = {Edinburgh},
%  booktitle    = "{Proc. EUSIPCO}",
%  month 	= sep,
%  pages 	= "776--779",
%  title 	= "On the performance of orthogonal source separation algorithms",
%  year 	= 1994}
%_________________________________________________________________________
% jade.m ends here
end