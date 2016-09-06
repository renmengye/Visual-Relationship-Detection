%   This file is for Predicting <subject, predicate, object> phrase and relationship

%   Distribution code Version 1.0 -- Copyright 2016, AI lab @ Stanford University.
%   
%   The Code is created based on the method described in the following paper 
%   [1] "Visual Relationship Detection with Language Priors",
%   Cewu Lu*, Ranjay Krishna*, Michael Bernstein, Li Fei-Fei, European Conference on Computer Vision, 
%   (ECCV 2016), 2016(oral). (* = indicates equal contribution)
%  
%   The code and the algorithm are for non-comercial use only.

[recall50P, recall100P, recall50R, recall100R] = ...
    relationshipPhraseDetection('/u/mren/pred.mat');
% fprintf('\n');
% fprintf('#######  Top recall results  ####### \n');
% fprintf('Phrase Det. R@50: %0.2f \n', 100 * recall50P);
% fprintf('Phrase Det. R@100: %0.2f \n', 100 * recall100P);
% fprintf('Relationship Det. R@50: %0.2f \n', 100 * recall50R);
% fprintf('Relationship Det. R@100: %0.2f \n', 100 * recall100R);
 
