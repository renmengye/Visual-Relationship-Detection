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
addpath('evaluation');
load('data/objectListN.mat'); 
% given a object category index and ouput the name of it.

load('data/obj2vec.mat'); 
% word-to-vector embeding based on https://github.com/danielfrg/word2vec
% input a word and ouput a vector.

load('data/UnionCNNfeaPredicate.mat')
% the CNN score on union of the boundingboxes of the two participating objects in that relationship. 
% we provide our scores (VGG based) here, but you can re-train a new model.

load('data/objectDetRCNN.mat');
% object detection results. The scores are mapped into [0,1]. 
% we provide detected object (RCCN with VGG) here, but you can use a better model (e.g. ResNet).
% three items: 
% detection_labels{k}: object category index in k^{th} testing image.
% detection_bboxes{k}: detected object bounding boxes in k^{th} testing image. 
% detection_confs{k}: confident score vector in k^{th} testing image. 

load('data/Wb.mat');
% W and b in Eq. (2) in [1]

%% We assume we have ground truth object detection
% we will change "predicate" in rlp_labels_ours use our prediction

load('evaluation/gt.mat');

ours = load('/u/mren/pred.mat');
%ours = load('/u/mren/output_valid_all.mat');
load('data/imagePath.mat');
image_ids = cellstr(ours.image_ids);
image_id_convert = zeros(length(image_ids), 1) - 1;
image_id_convert2 = zeros(length(imagePath), 1) - 1;
for ii = 1 : length(image_ids)
    found = 0;
    for jj = 1 : length(imagePath)
        if strcmp(imagePath{jj}, image_ids{ii})
            image_id_convert(ii) = jj;
            image_id_convert2(jj) = ii;
            found = 1;
            break;
        end
    end
    if ~found
        error(sprintf('Cannot find image id %s', image_ids{ii}))
    end
end

%% 
fprintf('\n');
fprintf('#######  Predicate computing Begins  ####### \n');
testNum = length(imagePath{ii});
rlp_labels_ours = cell(testNum, 1);
rlp_conf_ours = cell(testNum, 1);
sub_bboxes_ours = cell(testNum, 1);
obj_bboxes_ours = cell(testNum, 1);
rlp_labels_ours_dupe = cell(testNum, 1);
rlp_confs_ours_dupe = cell(testNum, 1);
sub_bboxes_ours_dupe = cell(testNum, 1);
obj_bboxes_ours_dupe = cell(testNum, 1);
for ii = 1 : length(imagePath)
    found = 0;
    for jj = 1 : length(image_ids)
        if strcmp(imagePath{ii}, image_ids{jj})
            found = 1;
            break
        end
    end
    if ~found
        rlp_labels_ours{ii} = [];
        rlp_confs_ours{ii} = [];
        rlp_labels_ours_dupe{ii} = [];
        rlp_confs_ours{ii} = [];
        sub_bboxes_ours_dupe{ii} = [];
        obj_bboxes_ours_dupe{ii} = [];
        sub_bboxes_ours{ii} = [];
        obj_bboxes_ours{ii} = [];
    end
end
for ii = 1 : size(image_id_convert, 1)
    ii2 = image_id_convert(ii);
    if mod(ii, 100) == 0
        fprintf([num2str(ii), 'th image is tested! \n']);
    end

    len = size(ours.data{ii}.labels, 1);
    rlp_confs_ours{ii2} = zeros(len, 1);
    % [len, 70] => [70, len] => [70 * len]
    rlp_confs_ours_dupe{ii2} = reshape(ours.data{ii}.conf', [70 * len, 1]);
    % [len, 70, 3] => [70, len, 3] => [70 * len, 3]
    rlp_labels_ours_dupe{ii2} = reshape(permute(...
        ours.data{ii}.labels, [2, 1, 3]), [70 * len, 3]);
    % Increment indexing by 1.
    rlp_labels_ours_dupe{ii2} = rlp_labels_ours_dupe{ii2} + 1;
    sub_bboxes_ours{ii2} = double(ours.data{ii}.subj_bbox);
    obj_bboxes_ours{ii2} = double(ours.data{ii}.obj_bbox);
    % [len, 70, 4] => [70, len, 4] => [70 * len, 4]
    sub_bboxes_ours_dupe{ii2} = reshape(repmat(reshape(...
        sub_bboxes_ours{ii2}, [1, len, 4]), [70, 1, 1]), [70 * len, 4]);
    obj_bboxes_ours_dupe{ii2} = reshape(repmat(reshape(...
        obj_bboxes_ours{ii2}, [1, len, 4]), [70, 1, 1]), [70 * len, 4]);
    
    for jj = 1 : len
        [m_score, m_pred] = max(ours.data{ii}.conf(jj, :));
        % Need to increment the ID by 1 because of MATLAB indexing.
        rlp_labels_ours{ii2}(jj, 1) = ours.data{ii}.labels(jj, m_pred, 1) + 1;
        rlp_labels_ours{ii2}(jj, 2) = m_pred;
        rlp_labels_ours{ii2}(jj, 3) = ours.data{ii}.labels(jj, m_pred, 3) + 1;
        rlp_confs_ours{ii2}(jj) = m_score;
    end
end

%% sort by confident score
for ii = 1 : length(rlp_confs_ours)
    [Confs, ind] = sort(rlp_confs_ours{ii}, 'descend');
    rlp_confs_ours{ii} = Confs;
    rlp_labels_ours{ii} = rlp_labels_ours{ii}(ind, :);
    sub_bboxes_ours{ii} = sub_bboxes_ours{ii}(ind, :);
    obj_bboxes_ours{ii} = obj_bboxes_ours{ii}(ind, :);
end

for ii = 1 : length(rlp_confs_ours_dupe)
    [Confs, ind] = sort(rlp_confs_ours_dupe{ii}, 'descend');
    rlp_confs_ours_dupe{ii} = Confs;
    rlp_confs_ours{ii}(1: min(5, size(rlp_confs_ours{ii}, 1)));
    rlp_labels_ours_dupe{ii} = rlp_labels_ours_dupe{ii}(ind, :);
    sub_bboxes_ours_dupe{ii} = sub_bboxes_ours_dupe{ii}(ind, :);
    obj_bboxes_ours_dupe{ii} = obj_bboxes_ours_dupe{ii}(ind, :);
end

%% computing Predicate Det. accuracy
fprintf('\n');
fprintf('#######  Top recall results (single vote) ####### \n');
recall50R = top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, ...
    sub_bboxes_ours, obj_bboxes_ours);
recall100R = top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, ...
    sub_bboxes_ours, obj_bboxes_ours);
fprintf('Predicate Det. R@50: %0.2f \n', 100 * recall50R);
fprintf('Predicate Det. R@100: %0.2f \n', 100 * recall100R);

%% computing Predicate Det. accuracy
fprintf('\n');
fprintf('#######  Top recall results (allow multi-vote) ####### \n');
recall50R = top_recall_Relationship(50, rlp_confs_ours_dupe, ...
    rlp_labels_ours_dupe, sub_bboxes_ours_dupe, obj_bboxes_ours_dupe);
recall100R = top_recall_Relationship(100, rlp_confs_ours_dupe, ...
    rlp_labels_ours_dupe, sub_bboxes_ours_dupe, obj_bboxes_ours_dupe);
fprintf('Predicate Det. R@50: %0.2f \n', 100 * recall50R);
fprintf('Predicate Det. R@100: %0.2f \n', 100 * recall100R);

