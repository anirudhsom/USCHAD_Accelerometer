function output = Chopping_inactivity(x,threshold_value)

if(nargin<2)
    threshold_value = 0.0005;
end
x = x(1:2:end,:);

%% Smoothing
for i = 1:size(x,2)
    sx(:,i) = smooth(x(:,i),5);
end

%% Chopping    
dx = diff(smooth(x(:,i),50));
indices = find(abs(dx)<threshold_value);
cx = sx;
cx(indices,:) = [];

% %
% figure;plot(sx)
% figure; plot(cx);
% %

output = cx;
end