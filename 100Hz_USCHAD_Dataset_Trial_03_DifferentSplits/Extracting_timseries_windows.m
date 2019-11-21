clc;
clear;
close all;

code_path = '/media/anirudh/Drive1/LLNL_Summer_2018/100Hz_USCHAD_Dataset';
dataset_path = code_path;

%% PARAMETERS
threshold_value = 0.0005;
window_size = 100; param.window_size = window_size;
min_window_size = window_size;
save_path = [code_path '/Window-size-' num2str(window_size)]; param.save_path = save_path;

%% MAIN CODE

if ~exist(save_path)
    mkdir(save_path);
end

failed_cases = [];
success_cases = [];
window_info = [];
dir_list = dir([dataset_path '/*.mat']);
zz = 1;
ff = 1;
ss = 1;
labels = [];

total_subjects = length(dir([code_path,'/time-series_data/Subject*']));

for i = 1:total_subjects
    
    param.subject_id =  i;

    folder_path = [code_path,'/time-series_data/Subject',num2str(i)];

    dir_list = dir([folder_path,'/a*']);
    
    for j = 1:length(dir_list)
        %j = 1
        load([folder_path,'/',dir_list(j).name]);
        x = sensor_readings(:,1:3);
        sx = Smoothing_activity(x);

        param.CLASS = str2num(activity_number);
        
        if length(sx)>=min_window_size
            %% Extracting time-series windows
            labels = Time_series_windows(sx,labels,param); % Function w.r.t. only this code
            success_cases(ss,:) = [param.subject_id  param.CLASS]; ss=ss+1;
        else
            failed_cases(ff,:) = [param.subject_id  param.CLASS]; ff=ff+1;
        end
    end
end

cd(code_path);
save([code_path '/Successful_Failed_cases_Window-size-' num2str(window_size) '.mat'],'failed_cases','success_cases');

save(['Labels-' num2str(window_size) '.mat'],'labels');