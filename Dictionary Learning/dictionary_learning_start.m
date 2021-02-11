img = rgb2gray(imread('c.jpg'));
X = double(img) / 255;
k = 50;
iter = 20;
tol = 1e-6;
[D,R] = dictionary_learning(X, k, iter, tol, 0);
imshow(D*R);
R = sparse(R);
