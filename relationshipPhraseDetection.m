function [recall50P, recall100P, recall50R, recall100R] = ...
    relationshipPhraseDetection(inputFile)

    addpath('evaluation');
    load('data/imagePath.mat');
    ours = load(inputFile);
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

    testNum = 1000;
    fprintf('#######  Relationship computing Begins  ####### \n');
    rlp_labels_ours = cell(testNum, 1);
    rlp_conf_ours = cell(testNum, 1);
    sub_bboxes_ours = cell(testNum, 1);
    obj_bboxes_ours = cell(testNum, 1);
    for ii = 1 : testNum
        rlp_labels_ours{ii} = [];
        rlp_confs_ours{ii} = []; 
        sub_bboxes_ours{ii} = []; 
        obj_bboxes_ours{ii} = [];
    end

    for ii = 1 : size(image_id_convert, 1)
        ii2 = image_id_convert(ii);

        if mod(ii, 100) == 0
            fprintf([num2str(ii), 'th image is tested! \n']);
        end
        
        len = size(ours.data{ii}.labels, 1);
        for jj = 1 : len
            [m_score, m_pred] = max(ours.data{ii}.conf(jj, :));
            rlp_confs_ours{ii2}(jj) = m_score;
            rlp_labels_ours{ii2}(jj, :) = ours.data{ii}.labels(jj, m_pred, :) + 1;
        end
        sub_bboxes_ours{ii2} = double(ours.data{ii}.subj_bbox);
        obj_bboxes_ours{ii2} = double(ours.data{ii}.obj_bbox);
    end

    %% sort by confident score
    for ii = 1 : length(rlp_confs_ours)
        [Confs, ind] = sort(rlp_confs_ours{ii}, 'descend');
        rlp_confs_ours{ii} = Confs;
        rlp_labels_ours{ii} = rlp_labels_ours{ii}(ind, :);
        sub_bboxes_ours{ii} = sub_bboxes_ours{ii}(ind, :);
        obj_bboxes_ours{ii} = obj_bboxes_ours{ii}(ind, :);
    end

    %% computing Phrase Det. and Relationship Det. accuracy
    fprintf('#######  Top recall results  ####### \n');
    recall50P = top_recall_Phrase(50, rlp_confs_ours, rlp_labels_ours, ...
                                  sub_bboxes_ours, obj_bboxes_ours); 
    recall100P = top_recall_Phrase(100, rlp_confs_ours, rlp_labels_ours, ...
                                   sub_bboxes_ours, obj_bboxes_ours);
    fprintf('Phrase Det. R@50: %0.2f \n', 100 * recall50P);
    fprintf('Phrase Det. R@100: %0.2f \n', 100 * recall100P);
    
    recall50R = top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, ...
                                        sub_bboxes_ours, obj_bboxes_ours);
    recall100R = top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, ...
                                         sub_bboxes_ours, obj_bboxes_ours);
    fprintf('Relationship Det. R@50: %0.2f \n', 100 * recall50R);
    fprintf('Relationship Det. R@100: %0.2f \n', 100 * recall100R);
end

