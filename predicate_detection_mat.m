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

ours = load('/u/mren/output_valid_top.mat');
load('data/imagePath.mat');
image_id_convert = zeros(length(image_ids), 1);
image_ids = cellstr(ours.image_ids);
for ii = 1 : length(image_ids)
    found = 0;
    for jj = 1 : length(imagePath)
        if strcmp(imagePath{jj}, image_ids{ii})
            image_id_convert(ii) = jj;
            found = 1;
            break;
        end
    end
    if ~found
        error(sprintf('Cannot find image id %s', image_ids{ii}))
    end
end
image_id_convert


% gt_tuple_label{j}(k,:) is a tuple that record categroy indexes of <subject, predicate, object>
% gt_sub_bboxes{j}: bounding boxes of subject 
% obj_bboxes_ours{j}: bounding boxes of object 

%% 
fprintf('\n');
fprintf('#######  Predicate computing Begins  ####### \n');
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
    rlp_confs_ours_dupe{ii2} = reshape(ours.data{ii}.conf, [len * 70, 1]);
    rlp_labels_ours_dupe{ii2} = reshape(ours.data{ii}.labels, [len * 70, 3]);
    sub_bboxes_ours{ii2} = double(ours.data{ii}.subj_bbox);
    obj_bboxes_ours{ii2} = double(ours.data{ii}.obj_bbox);
    sub_bboxes_ours_dupe{ii2} = reshape(repmat(reshape(...
        sub_bboxes_ours{ii2}, [1, len, 4]), [70, 1, 1]), [len * 70, 4]);
    obj_bboxes_ours_dupe{ii2} = reshape(repmat(reshape(...
        obj_bboxes_ours{ii2}, [1, len, 4]), [70, 1, 1]), [len * 70, 4]);
    
    cc = 1
    size(rlp_confs_ours_dupe{ii2})
    cc = 2
    size(sub_bboxes_ours_dupe{ii2})

    for jj = 1 : len
        [m_score, m_pred] = max(ours.data{ii}.conf(jj, :));
        rlp_labels_ours{ii2}(jj, 1) = ours.data{ii}.labels(jj, m_pred, 1);
        rlp_labels_ours{ii2}(jj, 2) = m_pred;
        rlp_labels_ours{ii2}(jj, 3) = ours.data{ii}.labels(jj, m_pred, 3);
        rlp_confs_ours{ii2}(jj) = m_score;
    end
    cc = 3
    size(rlp_confs_ours{ii2})
    cc = 4
    size(rlp_labels_ours{ii2})
    
end

%% sort by confident score

for ii = 1 : length(rlp_confs_ours)
    [Confs, ind] = sort(rlp_confs_ours{ii}, 'descend');
    rlp_confs_ours{ii} = Confs;
    rlp_labels_ours{ii} = rlp_labels_ours{ii}(ind,:);
    sub_bboxes_ours{ii} = sub_bboxes_ours{ii}(ind,:);
    obj_bboxes_ours{ii} = obj_bboxes_ours{ii}(ind,:);
end

for ii = 1 : length(rlp_confs_ours_dupe)
    [Confs, ind] = sort(rlp_confs_ours_dupe{ii}, 'descend');
    rlp_confs_ours_dupe{ii} = Confs;
    rlp_confs_ours{ii}(1: min(5, size(rlp_confs_ours{ii}, 1)));
    rlp_labels_ours_dupe{ii} = rlp_labels_ours_dupe{ii}(ind, :);
    sub_bboxes_ours_dupe{ii} = sub_bboxes_ours_dupe{ii}(ind, :);
    obj_bboxes_ours_dupe{ii} = obj_bboxes_ours_dupe{ii}(ind, :);
end

% save('results/predicate_det_result.mat', 'rlp_labels_ours', ...
%     'rlp_confs_ours', 'sub_bboxes_ours', 'obj_bboxes_ours');

%% computing Predicate Det. accuracy
fprintf('\n');
fprintf('#######  Top recall results (single vote) ####### \n');
recall100R = top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, ...
sub_bboxes_ours, obj_bboxes_ours);
recall50R = top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, ...
sub_bboxes_ours, obj_bboxes_ours);
fprintf('Predicate Det. R@100: %0.2f \n', 100*recall100R);
fprintf('Predicate Det. R@50: %0.2f \n', 100*recall50R);

%% computing Predicate Det. accuracy
fprintf('\n');
fprintf('#######  Top recall results (allow multi-vote) ####### \n');
recall100R = top_recall_Relationship(100, rlp_confs_ours_dupe, ...
rlp_labels_ours_dupe, sub_bboxes_ours_dupe, obj_bboxes_ours_dupe);
recall50R = top_recall_Relationship(50, rlp_confs_ours_dupe, ...
rlp_labels_ours_dupe, sub_bboxes_ours_dupe, obj_bboxes_ours_dupe);
fprintf('Predicate Det. R@100: %0.2f \n', 100 * recall100R);
fprintf('Predicate Det. R@50: %0.2f \n', 100 * recall50R);

fprintf('\n');
fprintf('#######  Zero-shot results  ####### \n');
zeroShot100R = zeroShot_top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
zeroShot50R = zeroShot_top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('zero-shot Predicate Det. R@100: %0.2f \n', 100*zeroShot100R);
fprintf('zero-shot Predicate Det. R@50: %0.2f \n', 100*zeroShot50R);

