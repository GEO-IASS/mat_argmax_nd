%%
b=argmax(5,1,int32(0));

%%
a = magic(3);
a
to = int32(0);
a = single(a);

[c0,b0,mb0] = argmaxcomp(a,0,to);c0
[c1,b1,mb1] = argmaxcomp(a,1,to);c1 
[c2,b2,mb2] = argmaxcomp(a,2,to);c2

%%
a = [1,2,3;4,5,6];
[c0,b0,mb0] = argmaxcomp(a,0,to);c0
[c1,b1,mb1] = argmaxcomp(a,1,to);c1 % operate by rows means that scan by columns
[c2,b2,mb2] = argmaxcomp(a,2,to);c2  % operate by cols means that scan by rows


%%
a = rand([3,1])
[c0,b0,mb0] = argmaxcomp(a,0,to);c0
[c1,b1,mb1] = argmaxcomp(a,1,to);c1
[c2,b2,mb2] = argmaxcomp(a,2,to);c2

%%
a = rand([1,3])
[c0,b0,mb0] = argmaxcomp(a,0,to);c0
[c1,b1,mb1] = argmaxcomp(a,1,to);c1
[c2,b2,mb2] = argmaxcomp(a,2,to);c2

%%
a = double(10*[0.4596    0.2180    0.7353]);
[c0,b0,mb0] = argmaxcomp(a,0,to);c0
[c1,b1,mb1] = argmaxcomp(a,1,to);c1
[c2,b2,mb2] = argmaxcomp(a,2,to);c2

%% Ksize/Q::csize < 4 ==>
a = int32(200:-1:100);
[c0,b0,mb0,dt0] = argmaxcomp(a,0,to);c0
b0
mb0
%%

%%
a = int32(255*rand([32,11]));
[c0,b0,mb0,dt0] = argmaxcomp(a,0,to);c0
[c1,b1,mb1,dt1] = argmaxcomp(a,1,to);c1
[c2,b2,mb2,dt2] = argmaxcomp(a,2,to);c2

%%
a = floor(10*rand([2,3,4]))

[c0,b0,mb0] = argmaxcomp(a,0,to);c0
[c1,b1,mb1] = argmaxcomp(a,1,to);c1
[c2,b2,mb2] = argmaxcomp(a,2,to);c2  
[c3,b3,mb3] = argmaxcomp(a,3,to);c3

%%
a = floor(10*rand([2,3,4,5]))

[c0,b0,mb0,dt0] = argmaxcomp(a,0,to);c0
[c1,b1,mb1,dt1] = argmaxcomp(a,1,to);c1
[c2,b2,mb2,dt2] = argmaxcomp(a,2,to);c2  
[c3,b3,mb3,dt3] = argmaxcomp(a,3,to);c3
[c4,b4,mb4,dt4] = argmaxcomp(a,4,to);c4


%%
a = logical(floor(10*rand([2,3,4])) > 5);

[c0,b0,mb0,dt0] = argmaxcomp(a,0,to);c0
[c1,b1,mb1,dt1] = argmaxcomp(a,1,to);c1
[c2,b2,mb2,dt2] = argmaxcomp(a,2,to);c2

%% Large
testlarge=1;
if testlarge
a = floor(10*rand([2,30,120,40,500]));
to=double(0);
[c0,b0,mb0,dt0] = argmaxcomp(a,0,to);c0
[c1,b1,mb1,dt1] = argmaxcomp(a,1,to);c1
[c2,b2,mb2,dt2] = argmaxcomp(a,2,to);c2  
[c3,b3,mb3,dt3] = argmaxcomp(a,3,to);c3
[c4,b4,mb4,dt4] = argmaxcomp(a,4,to);c4
[c5,b5,mb5,dt5] = argmaxcomp(a,5,to);c5
end


%%
A=single(rand(3000, 3000));
    to =int32(0);
[c0,b0,mb0,dt0,tx0] = argmaxcomp(A,0,to);c0
[c1,b1,mb1,dt1,tx1] = argmaxcomp(A,1,to);c1
[c2,b2,mb2,dt2,tx2] = argmaxcomp(A,2,to);c2

%%
[c0,b0,mb0,dt0,tx0] = argmaxcompcputime(A,0,double(0));
[c1,b1,mb1,dt1,tx1] = argmaxcompcputime(A,1,double(0));
[c2,b2,mb2,dt2,tx2] = argmaxcompcputime(A,2,double(0));

