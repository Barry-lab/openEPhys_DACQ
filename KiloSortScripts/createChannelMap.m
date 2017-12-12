function filename = createChannelMap(nchannels)
% Load 64 channels sample
load('chanMap64.mat')
% Change channel map
chanMap = chanMap(1:nchannels);
chanMap0ind = chanMap0ind(1:nchannels);
connected = connected(1:nchannels);
kcoords = kcoords(1:nchannels);
xcoords = xcoords(1:nchannels);
ycoords = ycoords(1:nchannels);
% Save channel map
filename = tempname;
save(filename, 'chanMap', 'chanMap0ind', 'connected', 'kcoords', 'xcoords', 'ycoords')
end

