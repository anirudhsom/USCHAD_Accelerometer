function output = Smoothing_activity(x)

x = x(1:2:end,:);

%% Smoothing
for i = 1:size(x,2)
    sx(:,i) = smooth(x(:,i),5);
end

output = sx;
end