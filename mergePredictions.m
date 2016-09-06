function mergePredictions(inputFiles, outputFile)
    % inputFiles: cell array of input filenames
    % outputFile: output filename
    
    ours_all = struct();
    num_models = length(inputFiles);
    for ii = 1 : num_models
        ours_ii = load(inputFiles{ii});
        if ~isfield(ours_all, 'image_ids')
            ours_all.image_ids = ours_ii.image_ids;
            ours_all.data = cell(num_models, 1);
            ours_all.data{ii} = ours_ii.data;
        else
            % Match image ID.
            % Convert from tmp to merge.
            num_images = length(ours_all.image_ids);
            ours_all.data{ii} = cell(num_images, 1);
            image_id_convert = zeros(num_images, 1) - 1;
            if length(ours_ii.image_ids) ~= num_images
                error('Image length does not match');
            end
            for jj = 1 : num_images
                found = 0;
                for kk = 1 : num_images
                    if strcmp(ours_ii.image_ids{jj}, ours_all.image_ids{kk})
                        image_id_convert(jj) = kk;
                        found = 1;
                        break;
                    end
                end
                if ~found
                    error(sprintf('Cannot find image id %s', ...
                    ours_ii.image_ids{jj}))
                end
            end
            for jj = 1 : length(ours_ii.image_ids)
                jj2 = image_id_convert(jj);
                num_pairs = size(ours_ii.data{jj}.labels, 1);
                pair_convert = zeros(num_pairs, 1);
                for kk = 1 : num_pairs
                    found = 0;
                    if size(ours_all.data{1}{jj2}.labels, 1) ~= num_pairs
                        error('Size does not match')
                    end
                    for kk2 = 1 : num_pairs
                        % subj1 = double([ours_ii.data{jj}.labels(kk, 1, 1), ...
                        % ours_ii.data{jj}.subj_bbox(kk, :)]);
                        % obj1 = double([ours_ii.data{jj}.labels(kk, 1, 3), ...
                        % ours_ii.data{jj}.obj_bbox(kk, :)]);
                        % subj2 = double([ours_all.data{1}{jj2}.labels(kk2, 1, 1), ...
                        % ours_all.data{1}{jj2}.subj_bbox(kk2, :)]);
                        % obj2 = double([ours_all.data{1}{jj2}.labels(kk2, 1, 3), ...
                        % ours_all.data{1}{jj2}.obj_bbox(kk2, :)]);
                        subj1 = double(ours_ii.data{jj}.subj_bbox(kk, :));
                        obj1 = double(ours_ii.data{jj}.obj_bbox(kk, :));
                        subj2 = double(ours_all.data{1}{jj2}.subj_bbox(kk2, :));
                        obj2 = double(ours_all.data{1}{jj2}.obj_bbox(kk2, :));
                        if norm(subj1 - subj2, 2) == 0 ...
                            && norm(obj1 - obj2, 2) == 0
                            found = 1;
                            break;
                        end
                    end
                    if ~found
                        error('Cannot find triplet match')
                    end
                    pair_convert(kk) = kk2;
                end

                ours_all.data{ii}{jj2} = struct();
                ours_all.data{ii}{jj2}.conf = ...
                    zeros(size(ours_ii.data{jj}.conf));
                num_pairs = size(ours_ii.data{jj}.labels, 1);
                for kk = 1 : num_pairs
                    kk2 = pair_convert(kk);
                    ours_all.data{ii}{jj2}.conf(kk2, :) = ...
                        ours_ii.data{jj}.conf(kk2, :);
                end 
            end
        end
    end

    ours_final = struct();
    ours_final.image_ids = ours_all.image_ids;
    ours_final.data = ours_all.data{1};
    for jj = 1 : num_images
        num_pairs = size(ours_final.data{jj}.labels, 1);
        conf = zeros(num_models, num_pairs, 70);
        for ii = 1 : num_models
            conf(ii, :, :) = ours_all.data{ii}{jj}.conf;
        end
        % conf = exp(conf);
        % conf = conf ./ repmat(sum(conf, 3), [1, 1, 70]);
        conf = reshape(conf(2, :, :), [num_pairs, 70]);
        % conf = reshape(mean(conf, 1), [num_pairs, 70]);
        ours_final.data{jj}.conf = conf;
    end
    save(outputFile, '-struct', 'ours_final');
end
