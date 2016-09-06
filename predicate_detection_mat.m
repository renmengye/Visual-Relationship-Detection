%   This file is for Predicting predicate 

%   Distribution code Version 1.0 -- Copyright 2016, AI lab @ Stanford University.
%   
%   The Code is created based on the method described in the following paper 
%   [1] "Visual Relationship Detection with Language Priors", 
%   Cewu Lu*, Ranjay Krishna*, Michael Bernstein, Li Fei-Fei, European Conference on Computer Vision, 
%   (ECCV 2016), 2016(oral). (* = indicates equal contribution)
%  
%   The code and the algorithm are for non-comercial use only.

%% data loading
%% computing Predicate Det. accuracy
% fprintf('\n');
[recall50R, recall100R, recall50RD, recall100RD] = ...
    predicateDetection('/u/mren/pred.mat')
% fprintf('#######  Top recall results (single vote) ####### \n');
% fprintf('Predicate Det. R@50: %0.2f \n', 100 * recall50R);
% fprintf('Predicate Det. R@100: %0.2f \n', 100 * recall100R);
% 
% fprintf('\n');
% fprintf('#######  Top recall results (allow multi-vote) ####### \n');
% fprintf('Predicate Det. R@50: %0.2f \n', 100 * recall50RD);
% fprintf('Predicate Det. R@100: %0.2f \n', 100 * recall100RD);

