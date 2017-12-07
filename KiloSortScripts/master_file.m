function [ varargout ] = master_file ( varargin )

if length(varargin) < 1
    nchannels = 64;
    pname = '/media/DataDrive/sander/Documents/KiloSortProcess';
    fname = 'experiment_1.dat';
elseif length(varargin) < 2
    nchannels = varargin{1};
    pname = '/media/DataDrive/sander/Documents/KiloSortProcess';
    fname = 'experiment_1.dat';
elseif length(varargin) < 3
    nchannels = varargin{1};
    pname = varargin{2};
    fname = 'experiment_1.dat';
else
    nchannels = varargin{1};
    pname = varargin{2};
    fname = varargin{3};
end
% default options are in parenthesis after the comment

addpath(genpath('/usr/local/MATLAB/R2017a/toolbox/kilosort')) % path to kilosort folder
addpath(genpath('/usr/local/MATLAB/R2017a/toolbox/npy-matlab')) % path to npy-matlab scripts

run('StandardConfig.m')

tic; % start timer
%
if ops.GPU     
    gpuDevice(1); % initialize GPU (will erase any existing GPU arrays)
end

if strcmp(ops.datatype , 'openEphys')
   ops = convertOpenEphysToRawBInary(ops);  % convert data, only for OpenEphys
end
%
[rez, DATA, uproj] = preprocessData(ops); % preprocess data and extract spikes for initialization
rez                = fitTemplates(rez, DATA, uproj);  % fit templates iteratively
rez                = fullMPMU(rez, DATA);% extract final spike times (overlapping extraction)

% AutoMerge. rez2Phy will use for clusters the new 5th column of st3 if you run this)
%     rez = merge_posthoc2(rez);

% save matlab results file
save(fullfile(ops.root,  'rez.mat'), 'rez', '-v7.3');

% save python results file for Phy
rezToPhy(rez, ops.root);

% remove temporary file
delete(ops.fproc);
%%
